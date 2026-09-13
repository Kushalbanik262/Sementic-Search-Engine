# Embedding Service - sentence embeddings over HTTP.
# Copyright (C) 2026 Kushal Banik
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Broker-free asyncio primitives the worker is built from.

`MicroBatcher` turns a trickle of small jobs into fewer, larger encodes.
`PrioritySlot` is the single encode slot of a process: torch already spreads
one encode across every core, so two concurrent encodes only thrash cache.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Generic, Hashable, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class Batch(Generic[T]):
    key: Hashable
    items: list[T]
    texts: int


@dataclass(slots=True)
class _Group(Generic[T]):
    first_added: float
    items: list[T] = field(default_factory=list)
    texts: int = 0


class MicroBatcher(Generic[T]):
    """Groups items by `key_of` and releases a group when it holds `max_texts`
    texts or its oldest item has waited `window_s`, whichever comes first.

    Items with different keys (for example query vs passage, or normalize on
    vs off) never share an encode, because they need different model calls.
    A single item larger than `max_texts` is released on its own.
    """

    def __init__(
        self,
        *,
        max_texts: int,
        window_s: float,
        size_of: Callable[[T], int],
        key_of: Callable[[T], Hashable],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_texts < 1:
            raise ValueError("max_texts must be at least 1")
        self._max_texts = max_texts
        self._window_s = max(window_s, 0.0)
        self._size_of = size_of
        self._key_of = key_of
        self._clock = clock
        self._groups: dict[Hashable, _Group[T]] = {}
        self._changed = asyncio.Event()

    def __len__(self) -> int:
        return sum(len(group.items) for group in self._groups.values())

    @property
    def pending_texts(self) -> int:
        return sum(group.texts for group in self._groups.values())

    def add(self, item: T) -> None:
        key = self._key_of(item)
        group = self._groups.get(key)
        if group is None:
            group = self._groups[key] = _Group(first_added=self._clock())
        group.items.append(item)
        group.texts += self._size_of(item)
        self._changed.set()

    def take_ready(self) -> Batch[T] | None:
        """Pop a batch that is due now, without waiting."""
        now = self._clock()
        due: tuple[float, Hashable] | None = None
        for key, group in self._groups.items():
            if group.texts >= self._max_texts:
                return self._pop(key)
            if now - group.first_added >= self._window_s:
                if due is None or group.first_added < due[0]:
                    due = (group.first_added, key)
        return self._pop(due[1]) if due else None

    async def next_batch(self) -> Batch[T]:
        """Wait for the next due batch."""
        while True:
            self._changed.clear()
            batch = self.take_ready()
            if batch is not None:
                return batch

            timeout = None
            if self._groups:
                oldest = min(group.first_added for group in self._groups.values())
                timeout = max(oldest + self._window_s - self._clock(), 0.0)
            try:
                await asyncio.wait_for(self._changed.wait(), timeout)
            except TimeoutError:
                pass

    def drain(self) -> list[Batch[T]]:
        """Pop everything regardless of size or age. Used on shutdown."""
        batches = []
        while self._groups:
            batches.append(self._pop(next(iter(self._groups))))
        return batches

    def _pop(self, key: Hashable) -> Batch[T]:
        group = self._groups.pop(key)
        taken: list[T] = []
        texts = 0
        # Always take at least one item, so an oversized one cannot wedge the group.
        while group.items and (not taken or texts + self._size_of(group.items[0]) <= self._max_texts):
            item = group.items.pop(0)
            taken.append(item)
            texts += self._size_of(item)

        if group.items:
            # Keep the original timestamp: leftovers are never held longer
            # than one window in total just because a batch filled up.
            group.texts -= texts
            self._groups[key] = group
            self._changed.set()
        return Batch(key=key, items=taken, texts=texts)


class PrioritySlot:
    """A one-holder lock where lower `priority` values are served first.

    Lets a worker running both lanes put the next query batch ahead of the
    next index batch. It never preempts: a query can still wait for the index
    batch already encoding, which `index_batch_max_texts` bounds.
    """

    def __init__(self) -> None:
        self._busy = False
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    @property
    def busy(self) -> bool:
        return self._busy

    @asynccontextmanager
    async def acquire(self, priority: int) -> AsyncIterator[None]:
        if self._busy or self._waiters:
            future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            heapq.heappush(self._waiters, (priority, next(self._seq), future))
            try:
                await future
            except asyncio.CancelledError:
                if future.done() and not future.cancelled():
                    # Handed the slot just as we were cancelled: pass it on.
                    self._release()
                else:
                    self._waiters = [w for w in self._waiters if w[2] is not future]
                    heapq.heapify(self._waiters)
                raise
        else:
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        while self._waiters:
            _, _, future = heapq.heappop(self._waiters)
            if not future.done():
                # Ownership moves straight to the waiter; `_busy` stays True.
                future.set_result(None)
                return
        self._busy = False
