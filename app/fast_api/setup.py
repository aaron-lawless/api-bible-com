from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional

import uvicorn

from pyrocket.logger_config import UvicornAccessMiddleware
from app.routes.ingest import ingest_router
from app.routes.search import search_router
from app.routes.documents import documents_router
from app.routes.admin import admin_router
from app.routes.ui import ui_router


class ContextDict(Dict[str, Any]):
    """
    Type hint for request context
    """
    request: Request
    response: Response

async def health_check():
    return {"status": "ok"}

async def get_context(request: Request, response: Response) -> ContextDict:
    """
    Get request context for REST API endpoints
    """
    return {
        "request": request,
        "response": response,
    }

def register_routes(app: FastAPI):
    """
    Register API routes for the application
    """
    # Health check route (for railway deployment)
    app.add_api_route("/health", health_check, include_in_schema=False)  

    # Register other routes
    app.include_router(ui_router)
    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(documents_router)
    app.include_router(admin_router)


def register_middleware(app: FastAPI):
    """
    Register middleware for the FastAPI application
    """
    app.add_middleware(UvicornAccessMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], # TODO: Configure this properly in production (might be fine as public apis)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

async def start_app(lifespan: callable):
    app_dir = Path(__file__).resolve().parent.parent

    app = FastAPI(
        title="SacredScript API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Mounting the styling at the parent level for the UI and admin pages
    app.mount(
        "/static",
        StaticFiles(directory=str(app_dir / "static")),
        name="static",
    )

    register_middleware(app)
    register_routes(app)
    config = uvicorn.Config(app, host='0.0.0.0', port=8080)
    server = uvicorn.Server(config)
    await server.serve()