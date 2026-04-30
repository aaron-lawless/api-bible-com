import uuid
from unittest.mock import patch

import openai
import pytest


MOCK_ANSWER = {
    "answer": "The answer is found in the provided excerpts.",
    "sources": [
        {
            "document_id": str(uuid.uuid4()),
            "title": "Test Book",
            "author": "Author Name",
            "chunk_index": 0,
            "content": "Relevant chunk content here.",
            "score": 0.95,
        }
    ],
}


# ── Input validation ─────────────────────────────────────────────────────────


def test_search_missing_body(client):
    resp = client.post("/search", content="", headers={"content-type": "application/json"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_search_missing_query(client):
    resp = client.post("/search", json={"top_k": 5})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_search_invalid_top_k(client):
    resp = client.post("/search", json={"query": "What is truth?", "top_k": -1})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_search_invalid_document_ids(client):
    resp = client.post(
        "/search",
        json={"query": "What is truth?", "document_ids": ["not-a-uuid"]},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_search_document_ids_not_a_list(client):
    resp = client.post(
        "/search",
        json={"query": "What?", "document_ids": "single-string"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


# ── Success path ─────────────────────────────────────────────────────────────


def test_search_success(client):
    with patch(
        "app.routes.search.answer_question", return_value=MOCK_ANSWER
    ) as mock_answer:
        resp = client.post("/search", json={"query": "What is the answer?", "top_k": 5})

    assert resp.status_code == 200
    result = resp.json()
    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["sources"], list)
    mock_answer.assert_called_once()


def test_search_with_document_ids(client):
    doc_id = str(uuid.uuid4())
    with patch(
        "app.routes.search.answer_question", return_value=MOCK_ANSWER
    ) as mock_answer:
        resp = client.post(
            "/search",
            json={"query": "What is the answer?", "document_ids": [doc_id]},
        )

    assert resp.status_code == 200
    _call_kwargs = mock_answer.call_args.kwargs
    assert _call_kwargs["document_ids"] == [doc_id]


# ── External-service error paths ─────────────────────────────────────────────


def test_search_openai_failure(client):
    with patch(
        "app.routes.search.answer_question",
        side_effect=openai.APIConnectionError(request=None),
    ):
        resp = client.post("/search", json={"query": "What is the answer?"})

    assert resp.status_code == 502
    assert "error" in resp.json()


def test_search_unexpected_error(client):
    with patch(
        "app.routes.search.answer_question",
        side_effect=RuntimeError("something broke"),
    ):
        resp = client.post("/search", json={"query": "What is the answer?"})

    assert resp.status_code == 500
    assert "error" in resp.json()

