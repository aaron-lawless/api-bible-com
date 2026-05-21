# import uvicorn

# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     try:
#         uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
#     except KeyboardInterrupt:
#         pass


from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Code to run before the application starts
    print("Starting up...")

    yield

    # Code to run after the application shuts down
    print("Shutting down...")
