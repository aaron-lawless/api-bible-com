import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()


def _running_on_railway() -> bool:
    return any(
        os.environ.get(name)
        for name in ("RAILWAY_ENVIRONMENT", "RAILWAY_PROJECT_ID", "RAILWAY_SERVICE_ID")
    )


def _looks_like_railway_internal_url(database_url: str) -> bool:
    hostname = urlparse(database_url).hostname or ""
    return hostname.endswith(".railway.internal")

# -- Database URI resolution logic: local override is sqlite
def _resolve_database_uri() -> str:
    local_database_url = os.environ.get("LOCAL_DATABASE_URL")
    if local_database_url:
        return local_database_url

    database_url = os.environ.get("DATABASE_URL")
    if database_url and not (
        _looks_like_railway_internal_url(database_url) and not _running_on_railway()
    ):
        return database_url

    return "sqlite:///aibible-dev.db"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-me")
    SQLALCHEMY_DATABASE_URI = _resolve_database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 52428800))
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
