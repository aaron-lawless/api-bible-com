from unittest.mock import patch

import pytest

from app.config import _looks_like_railway_internal_url, _resolve_database_uri


def test_resolve_database_uri_prefers_local_override(monkeypatch):
    monkeypatch.setenv("LOCAL_DATABASE_URL", "sqlite:///custom-dev.db")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:secret@postgres.railway.internal:5432/railway",
    )

    assert _resolve_database_uri() == "sqlite:///custom-dev.db"


def test_resolve_database_uri_falls_back_for_local_railway_internal(monkeypatch):
    monkeypatch.delenv("LOCAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RAILWAY_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_ID", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:secret@postgres.railway.internal:5432/railway",
    )

    assert _resolve_database_uri() == "sqlite:///aibible-dev.db"


def test_resolve_database_uri_keeps_database_url_on_railway(monkeypatch):
    database_url = "postgresql://postgres:secret@postgres.railway.internal:5432/railway"
    monkeypatch.delenv("LOCAL_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")

    assert _resolve_database_uri() == database_url


def test_looks_like_railway_internal_url():
    assert _looks_like_railway_internal_url(
        "postgresql://postgres:secret@postgres.railway.internal:5432/railway"
    )
    assert not _looks_like_railway_internal_url(
        "postgresql://user:pass@localhost:5432/aibible"
    )


def test_verify_database_connection_raises_on_failure():
    from app.database import verify_database_connection, engine

    with patch.object(engine, "connect", side_effect=Exception("db down")):
        with pytest.raises(RuntimeError, match="Database connection failed during startup"):
            verify_database_connection()
