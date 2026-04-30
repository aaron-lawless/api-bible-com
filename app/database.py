import logging
from urllib.parse import urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from app.config import Config

logger = logging.getLogger("db_startup")

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _format_database_target(database_uri: str) -> str:
    parsed = urlparse(database_uri)
    scheme = parsed.scheme or "database"

    if scheme.startswith("postgres"):
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database_name = (parsed.path or "/").lstrip("/") or "unknown-db"
        return f"Postgres host={host} port={port} db={database_name}"

    if scheme.startswith("sqlite"):
        return f"SQLite {parsed.path or database_uri}"

    return f"{scheme} {database_uri}"


def verify_database_connection() -> None:
    logger.info(
        "Connecting to %s", _format_database_target(Config.SQLALCHEMY_DATABASE_URI)
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connection established")
    except Exception as exc:
        logger.exception("Database connection failed during startup")
        raise RuntimeError("Database connection failed during startup") from exc
