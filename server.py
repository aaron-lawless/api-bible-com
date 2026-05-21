# import uvicorn
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.fast_api.prometheus import start_prometheus_app
from app.fast_api.setup import start_app
from db.database import close_database_connections, verify_database_connection
from pyrocket.logger_config import configure_access_logging, configure_app_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Configure logging for application startup
    logger = configure_app_logging()
    configure_access_logging()
    logger.info("SacredScript API starting up with structured logging enabled.")

    # Initialize PostgreSQL connection
    try:
        verify_database_connection()
        logger.info("Database connection verified successfully.")
    except Exception as e:
        logger.warning("Failed to verify database connection during startup: %s", e)
        # TODO: need to have this fail the startup if the database connection cannot be established, but for now we just log it and continue starting up so that the UI can show the error message instead of the whole app failing to start

    yield

    # Close PostgreSQL connection and perform cleanup
    try:
        close_database_connections()
        logger.info("Database connections closed successfully.")
    except Exception as e:
        logger.warning("Failed to close database connections during shutdown: %s", e)

    logger.info("SacredScript API shutting down.")

async def main():
    try:
        await asyncio.gather(start_app(lifespan), start_prometheus_app())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except asyncio.CancelledError:
        print("Services cancelled, shutting down...")

if __name__ == "__main__":
    print("Starting SacredScript API...")
    print ("  API docs:  http://localhost:8080/docs")
    print ("  Search UI: http://localhost:8080/ui")
    print (" Metrics:    http://localhost:9090/metrics")
    print (" Admin UI:   http://localhost:8080/admin")
    print (" Health Check: http://localhost:8080/health")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Clean shutdown complete!")



