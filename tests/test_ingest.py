import io
import uuid
from unittest.mock import MagicMock, patch

import openai
import pytest


# ── Input validation tests (no mocking needed) ──────────────────────────────


def test_ingest_missing_file(client):
    resp = client.post("/ingest", data={"title": "Test"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_ingest_missing_title(client):
    resp = client.post(
        "/ingest",
        files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_ingest_unsupported_file_type(client):
    resp = client.post(
        "/ingest",
        files={"file": ("report.xlsx", io.BytesIO(b"data"), "application/octet-stream")},
        data={"title": "Test Doc"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_ingest_invalid_date_published(client):
    with (
        patch("app.routes.ingest.extract_text", return_value="text"),
        patch(
            "app.routes.ingest.chunk_text",
            return_value=[{"content": "chunk", "chunk_index": 0}],
        ),
        patch("app.routes.ingest.embed_chunks", return_value=[[0.1] * 1536]),
    ):
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test", "date_published": "not-a-date"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


# ── Success path ─────────────────────────────────────────────────────────────


def test_ingest_success(client, mock_db):
    doc_id = uuid.uuid4()

    mock_doc = MagicMock()
    mock_doc.id = doc_id

    with (
        patch("app.routes.ingest.extract_text", return_value="Sample document text"),
        patch(
            "app.routes.ingest.chunk_text",
            return_value=[{"content": "chunk text", "chunk_index": 0}],
        ),
        patch("app.routes.ingest.embed_chunks", return_value=[[0.1] * 1536]),
        patch("app.routes.ingest.Document", return_value=mock_doc),
        patch("app.routes.ingest.chunk_store") as mock_chunk_store,
    ):
        mock_chunk_store.store_chunks.return_value = None

        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            data={"title": "Test Document", "author": "Test Author"},
        )

    assert resp.status_code == 201
    result = resp.json()
    assert result["document_id"] == str(doc_id)
    assert result["chunk_count"] == 1
    assert result["title"] == "Test Document"


# ── External-service error paths ─────────────────────────────────────────────


def test_ingest_openai_failure(client):
    with (
        patch("app.routes.ingest.extract_text", return_value="text"),
        patch(
            "app.routes.ingest.chunk_text",
            return_value=[{"content": "chunk", "chunk_index": 0}],
        ),
        patch(
            "app.routes.ingest.embed_chunks",
            side_effect=openai.APIConnectionError(request=MagicMock()),
        ),
    ):
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test"},
        )

    assert resp.status_code == 502
    assert "error" in resp.json()


def test_ingest_db_failure(client, mock_db):
    doc_id = uuid.uuid4()
    mock_doc = MagicMock()
    mock_doc.id = doc_id
    mock_db.flush.side_effect = Exception("DB connection lost")

    with (
        patch("app.routes.ingest.extract_text", return_value="text"),
        patch(
            "app.routes.ingest.chunk_text",
            return_value=[{"content": "chunk", "chunk_index": 0}],
        ),
        patch("app.routes.ingest.embed_chunks", return_value=[[0.1] * 1536]),
        patch("app.routes.ingest.Document", return_value=mock_doc),
        patch("app.routes.ingest.chunk_store"),
    ):
        resp = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test"},
        )

    assert resp.status_code == 500
    assert "error" in resp.json()

