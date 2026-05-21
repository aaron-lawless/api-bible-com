from unittest.mock import patch

import pytest

from app.config import Mode, _resolve_mode, _resolve_database_uri


def test_resolve_mode_defaults_to_local(monkeypatch):
    monkeypatch.delenv("MODE", raising=False)
    assert _resolve_mode() == Mode.LOCAL


def test_resolve_mode_nprd(monkeypatch):
    monkeypatch.setenv("MODE", "nprd")
    assert _resolve_mode() == Mode.NPRD


def test_resolve_mode_prd(monkeypatch):
    monkeypatch.setenv("MODE", "prd")
    assert _resolve_mode() == Mode.PRD


def test_resolve_mode_invalid(monkeypatch):
    monkeypatch.setenv("MODE", "staging")
    with pytest.raises(ValueError, match="Invalid MODE"):
        _resolve_mode()


def test_resolve_database_uri_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        _resolve_database_uri(Mode.LOCAL)


def test_resolve_database_uri_nprd_uses_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    assert _resolve_database_uri(Mode.NPRD) == "postgresql://user:pass@host:5432/db"


def test_resolve_database_uri_prd_uses_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@host:5432/db")
    assert _resolve_database_uri(Mode.PRD) == "postgresql://user:pass@host:5432/db"


def test_resolve_database_uri_nprd_raises_without_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        _resolve_database_uri(Mode.NPRD)


def test_verify_database_connection_raises_on_failure():
    from db.database import verify_database_connection, engine

    with patch.object(engine, "connect", side_effect=Exception("db down")):
        with pytest.raises(RuntimeError, match="Database connection failed during startup"):
            verify_database_connection()
