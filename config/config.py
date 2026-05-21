import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()


class Mode(str, Enum):
    LOCAL = "local"
    NPRD = "nprd"
    PRD = "prd"


def _resolve_mode() -> Mode:
    raw = os.environ.get("MODE", "local").lower()
    try:
        return Mode(raw)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise ValueError(f"Invalid MODE '{raw}'. Must be one of: {valid}")


def _resolve_database_uri(mode: Mode) -> str:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")
    return database_url


_mode = _resolve_mode()


class Config:
    MODE: Mode = _mode
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    SQLALCHEMY_DATABASE_URI = _resolve_database_uri(_mode)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o")
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 52428800))
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
    SCRAPER_TIMEOUT = int(os.environ.get("SCRAPER_TIMEOUT", 30))
    TIER1_PREFILTER_TOP_K = int(os.environ.get("TIER1_PREFILTER_TOP_K", 20))
