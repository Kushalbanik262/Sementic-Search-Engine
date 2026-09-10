"""Semantic quality checks.

These guard the thing unit tests miss: the service can return well-shaped
vectors that are useless for search. A model swap, a lost query instruction
or a bad pooling change breaks these while every other test stays green.

Thresholds are deliberately loose -- they catch things being broken, not
small score drift.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .conftest import cosine, embed

DOCUMENTS = [
    "To reset your password, open Settings and click Forgot Password.",
    "Our office is closed on public holidays and weekends.",
    "The mitochondria is the powerhouse of the cell.",
    "Refunds are issued to the original payment method within 5 days.",
]


@pytest.fixture(scope="module")
def document_vectors(client: TestClient) -> list[list[float]]:
    return embed(client, DOCUMENTS, input_type="passage")["embeddings"]


def _search(client: TestClient, query: str, document_vectors) -> int:
    """Return the index of the best-matching document."""
    query_vector = embed(client, query, input_type="query")["embeddings"][0]
    scores = [cosine(query_vector, doc) for doc in document_vectors]
    return max(range(len(scores)), key=scores.__getitem__)


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        ("how do I change my password", 0),
        ("are you open on Christmas day", 1),
        ("what part of a cell produces energy", 2),
        ("when will I get my money back", 3),
    ],
)
def test_queries_retrieve_the_right_document(
    client: TestClient, document_vectors, query: str, expected_index: int
) -> None:
    assert _search(client, query, document_vectors) == expected_index


def test_identical_text_scores_near_one(client: TestClient) -> None:
    text = "the quick brown fox"
    body = embed(client, [text, text])
    assert cosine(*body["embeddings"]) == pytest.approx(1.0, abs=1e-5)


def test_paraphrases_beat_unrelated_text(client: TestClient) -> None:
    body = embed(
        client,
        [
            "The cat sat on the mat.",
            "A feline was resting on the rug.",
            "Quarterly revenue grew by twelve percent.",
        ],
    )
    cat, paraphrase, unrelated = body["embeddings"]
    assert cosine(cat, paraphrase) > cosine(cat, unrelated)


def test_query_instruction_widens_the_separation_margin(
    client: TestClient, document_vectors
) -> None:
    """The whole point of the query/passage split.

    Measured on margin -- the correct document's score minus the best
    distractor's -- rather than on raw cosine. The instruction shifts the
    entire score distribution upward, so absolute similarities are not
    comparable between the two encodings; only the separation between right
    and wrong answers is. Averaged over several queries because any single
    pair can move either way.
    """
    keyword_queries = [("password", 0), ("holiday hours", 1), ("refund", 3)]

    def mean_margin(input_type: str) -> float:
        margins = []
        for query, target_index in keyword_queries:
            vector = embed(client, query, input_type=input_type)["embeddings"][0]
            scores = [cosine(vector, doc) for doc in document_vectors]
            best_distractor = max(
                score for index, score in enumerate(scores) if index != target_index
            )
            margins.append(scores[target_index] - best_distractor)
        return sum(margins) / len(margins)

    assert mean_margin("query") > mean_margin("passage")


def test_truncated_documents_lose_their_tail(client: TestClient) -> None:
    """Shows why the `truncated` flag matters rather than just reporting it.

    Content pushed past the token limit does not influence the embedding, so
    a long document silently stops matching queries about its later sections.
    """
    filler = "The weather in London is mild and often overcast. " * 120
    tail = " Our refund policy allows returns within thirty days."

    long_document = embed(client, filler + tail)
    assert long_document["truncated"] == [True]

    refund_query = embed(client, "refund policy", input_type="query")["embeddings"][0]
    weather_query = embed(client, "London weather", input_type="query")["embeddings"][0]
    vector = long_document["embeddings"][0]

    # The tail was cut, so the document matches its surviving head instead.
    assert cosine(weather_query, vector) > cosine(refund_query, vector)
