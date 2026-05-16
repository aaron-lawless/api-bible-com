import io
import uuid
from unittest.mock import MagicMock, patch

import openai
import pytest


# ?? /ingest/pdf ??????????????????????????????????????????????????????????????


def test_pdf_missing_file(client):
    resp = client.post("/ingest/pdf", data={"title": "Test", "summary": "A summary"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_pdf_missing_title(client):
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
        data={"summary": "A summary"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_pdf_missing_summary(client):
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
        data={"title": "Test"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_pdf_unsupported_file_type(client):
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("report.xlsx", io.BytesIO(b"data"), "application/octet-stream")},
        data={"title": "Test Doc", "summary": "A summary"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_pdf_invalid_date_published(client):
    with (
        patch("app.routes.ingest.extract_pages", return_value=[(1, "page text")]),
        patch("app.routes.ingest.embed_text", return_value=[0.1] * 1536),
    ):
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test", "summary": "A summary", "date_published": "not-a-date"},
        )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_pdf_success(client, mock_db):
    doc_id = uuid.uuid4()
    mock_doc = MagicMock()
    mock_doc.document_id = doc_id
    mock_db.flush.return_value = None

    with (
        patch("app.routes.ingest.extract_pages", return_value=[(1, "page text")]),
        patch("app.routes.ingest.embed_text", return_value=[0.1] * 1536),
        patch("app.routes.ingest.Document", return_value=mock_doc),
    ):
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            data={"title": "Test Document", "author": "Test Author", "summary": "A summary"},
        )

    assert resp.status_code == 201
    result = resp.json()
    assert result["document_id"] == str(doc_id)
    assert result["total_pages"] == 1
    assert result["title"] == "Test Document"


def test_pdf_openai_failure(client):
    with (
        patch("app.routes.ingest.extract_pages", return_value=[(1, "text")]),
        patch(
            "app.routes.ingest.embed_text",
            side_effect=openai.APIConnectionError(request=MagicMock()),
        ),
    ):
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test", "summary": "A summary"},
        )
    assert resp.status_code == 502
    assert "error" in resp.json()


def test_pdf_db_failure(client, mock_db):
    mock_doc = MagicMock()
    mock_doc.document_id = uuid.uuid4()
    mock_db.flush.side_effect = Exception("DB connection lost")

    with (
        patch("app.routes.ingest.extract_pages", return_value=[(1, "text")]),
        patch("app.routes.ingest.embed_text", return_value=[0.1] * 1536),
        patch("app.routes.ingest.Document", return_value=mock_doc),
    ):
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"title": "Test", "summary": "A summary"},
        )
    assert resp.status_code == 500
    assert "error" in resp.json()


# ?? /ingest/url ???????????????????????????????????????????????????????????????


def test_url_missing_url(client):
    resp = client.post("/ingest/url", data={"title": "Test", "summary": "A summary"})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_url_missing_title(client):
    resp = client.post(
        "/ingest/url",
        data={"url": "https://example.com/article", "summary": "A summary"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_url_missing_summary(client):
    resp = client.post(
        "/ingest/url",
        data={"url": "https://example.com/article", "title": "Test"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_url_scrape_failure(client):
    with patch(
        "app.routes.ingest.scrape_url",
        side_effect=ValueError("HTTP 404"),
    ):
        resp = client.post(
            "/ingest/url",
            data={
                "url": "https://example.com/missing",
                "title": "Test",
                "summary": "A summary",
            },
        )
    assert resp.status_code == 422


def test_url_success(client, mock_db):
    doc_id = uuid.uuid4()
    mock_doc = MagicMock()
    mock_doc.document_id = doc_id
    mock_db.flush.return_value = None

    with (
        patch("app.routes.ingest.scrape_url", return_value="Article body text here."),
        patch("app.routes.ingest.embed_text", return_value=[0.1] * 1536),
        patch("app.routes.ingest.Document", return_value=mock_doc),
    ):
        resp = client.post(
            "/ingest/url",
            data={
                "url": "https://example.com/article",
                "title": "Test Article",
                "summary": "A summary",
            },
        )

    assert resp.status_code == 201
    result = resp.json()
    assert result["document_id"] == str(doc_id)
    assert result["title"] == "Test Article"




