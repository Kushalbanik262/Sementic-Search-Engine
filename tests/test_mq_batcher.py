"""Micro-batcher and encode slot, with no broker and no model."""

from __future__ import annotations

import asyncio

import pytest

from mq.batcher import MicroBatcher, PrioritySlot

pytestmark = pytest.mark.anyio


def make(max_texts: int = 10, window_s: float = 0.05) -> MicroBatcher[tuple[str, int]]:
    # items are (key, text_count)
    return MicroBatcher(max_texts=max_texts, window_s=window_s, size_of=lambda i: i[1], key_of=lambda i: i[0])


async def test_flushes_when_full_without_waiting_for_the_window() -> None:
    batcher = make(max_texts=4, window_s=60)
    batcher.add(("p", 2))
    batcher.add(("p", 2))
    batch = await asyncio.wait_for(batcher.next_batch(), 0.5)
    assert batch.texts == 4 and len(batch.items) == 2


async def test_flushes_a_partial_batch_after_the_window() -> None:
    batcher = make(max_texts=100, window_s=0.02)
    batcher.add(("p", 1))
    loop = asyncio.get_running_loop()
    started = loop.time()
    batch = await asyncio.wait_for(batcher.next_batch(), 0.5)
    assert batch.texts == 1
    assert loop.time() - started >= 0.015


async def test_never_mixes_keys() -> None:
    batcher = make(max_texts=100, window_s=0)
    batcher.add(("query", 1))
    batcher.add(("passage", 1))
    batcher.add(("query", 1))
    first = await batcher.next_batch()
    second = await batcher.next_batch()
    assert {first.key, second.key} == {"query", "passage"}
    assert all(item[0] == batch.key for batch in (first, second) for item in batch.items)


async def test_splits_at_max_texts_and_leftovers_go_next() -> None:
    batcher = make(max_texts=5, window_s=0.05)
    for _ in range(4):
        batcher.add(("p", 2))
    first = await asyncio.wait_for(batcher.next_batch(), 0.5)
    assert first.texts == 4  # a third item of 2 would exceed 5
    # leftovers keep their original timestamp, so they wait out the rest of
    # the first window rather than starting a new one
    second = await asyncio.wait_for(batcher.next_batch(), 0.5)
    assert second.texts == 4
    assert len(batcher) == 0


async def test_oversized_item_goes_alone() -> None:
    batcher = make(max_texts=3, window_s=60)
    batcher.add(("p", 50))
    batch = await asyncio.wait_for(batcher.next_batch(), 0.5)
    assert batch.texts == 50


async def test_waiter_wakes_when_an_item_arrives() -> None:
    batcher = make(max_texts=1, window_s=60)
    waiter = asyncio.create_task(batcher.next_batch())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    batcher.add(("p", 1))
    batch = await asyncio.wait_for(waiter, 0.5)
    assert batch.texts == 1


async def test_drain_returns_everything() -> None:
    batcher = make(max_texts=2, window_s=60)
    for key in ("a", "a", "a", "b"):
        batcher.add((key, 1))
    batches = batcher.drain()
    assert sum(len(b.items) for b in batches) == 4
    assert len(batcher) == 0


async def test_priority_slot_serves_lower_priority_value_first() -> None:
    slot = PrioritySlot()
    order: list[str] = []

    async def use(name: str, priority: int) -> None:
        async with slot.acquire(priority):
            order.append(name)
            await asyncio.sleep(0.01)

    async with slot.acquire(1):  # an index batch is encoding
        tasks = [
            asyncio.create_task(use("index-2", 1)),
            asyncio.create_task(use("index-3", 1)),
        ]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(use("query", 0)))
        await asyncio.sleep(0.01)
    await asyncio.gather(*tasks)

    assert order[0] == "query"
    assert not slot.busy


async def test_priority_slot_survives_a_cancelled_waiter() -> None:
    slot = PrioritySlot()
    async with slot.acquire(0):
        waiter = asyncio.create_task(slot.acquire(0).__aenter__())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    assert not slot.busy
    async with slot.acquire(0):
        assert slot.busy
