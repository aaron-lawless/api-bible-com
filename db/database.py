import logging
from urllib.parse import urlparse

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

from config.config import Config

logger = logging.getLogger("db_startup")

engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Register the pgvector extension on every new connection so that
# SQLAlchemy's metadata operations (create_all, etc.) can see vector columns.
@event.listens_for(engine, "connect")
def _register_vector_extension(dbapi_conn, _connection_record):
    with dbapi_conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    dbapi_conn.commit()

# SQLAlchemy base class for models to inherit from -- This is required for the db models
class Base(DeclarativeBase):
    pass

# Dependency for getting a database session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper function to format the database target for logging at startup, with special handling for Postgres URIs
def _format_database_target(database_uri: str) -> str:
    parsed = urlparse(database_uri)
    scheme = parsed.scheme or "database"

    if scheme.startswith("postgres"):
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database_name = (parsed.path or "/").lstrip("/") or "unknown-db"
        return f"Postgres host={host} port={port} db={database_name}"

    return f"{scheme} {database_uri}"

# Helper function to verify database connectivity at application startup, with logging
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


def close_database_connections() -> None:
    """Dispose the SQLAlchemy engine and close pooled database connections."""
    logger.info("Disposing database engine and closing pooled connections")
    try:
        engine.dispose()
        logger.info("Database engine disposed successfully")
    except Exception as exc:
        logger.exception("Failed to dispose database engine")
        raise RuntimeError("Failed to close database connections") from exc
