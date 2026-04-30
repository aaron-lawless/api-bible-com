import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_APP_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import verify_database_connection

    verify_database_connection()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Bible",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        SessionMiddleware,
        secret_key=Config.SECRET_KEY,
    )

    app.mount(
        "/static",
        StaticFiles(directory=str(_APP_DIR / "static")),
        name="static",
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content={"error": exc.detail}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        msg = errors[0]["msg"] if errors else "Validation error"
        return JSONResponse(status_code=400, content={"error": msg})

    from app.routes import register_routers

    register_routers(app)

    _templates = Jinja2Templates(directory=str(_APP_DIR / "templates"))

    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        return {"status": "ok"}

    # Auto-create tables for local SQLite dev
    if Config.SQLALCHEMY_DATABASE_URI.startswith("sqlite:///"):
        from app.database import Base, engine

        Base.metadata.create_all(bind=engine)

    return app

