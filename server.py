# import uvicorn

# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     try:
#         uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
#     except KeyboardInterrupt:
#         pass


import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from db.database import close_database_connections, verify_database_connection
from logging.logger_config import configure_access_logging, configure_app_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Configure logging for application startup
    logger = configure_app_logging()
    configure_access_logging()
    logger.info("API Bible starting up with structured logging enabled.")

    # Initialize PostgreSQL connection
    try:
        await verify_database_connection()
        logger.info("Database connection verified successfully.")
    except Exception as e:
        logger.warning("Failed to verify database connection during startup: %s", e)
        # TODO: need to have this fail the startup if the database connection cannot be established, but for now we just log it and continue starting up so that the UI can show the error message instead of the whole app failing to start

    yield

    # Close PostgreSQL connection and perform cleanup
    try:
        await close_database_connections()
        logger.info("Database connections closed successfully.")
    except Exception as e:
        logger.warning("Failed to close database connections during shutdown: %s", e)

    logger.info("API Bible shutting down.")

async def main():
    try:
        await asyncio.gather(start_app(lifespan), start_premetheus(), start_health_check())
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    except asyncio.CancelledError:
        print("Services cancelled, shutting down...")

if __name__ == "__main__":
    print("Starting API Bible...")
    print ("  API docs:  http://localhost:8000/docs")
    print ("  Search UI: http://localhost:8000/ui")
    print (" Metrics:    http://localhost:8000/metrics")
    print (" Admin UI:   http://localhost:8000/admin")
    print (" Health Check: http://localhost:8000/health")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Clean shutdown complete!")



