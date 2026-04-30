from fastapi import FastAPI


def register_routers(app: FastAPI) -> None:
    from app.routes.ingest import ingest_router
    from app.routes.search import search_router
    from app.routes.documents import documents_router
    from app.routes.admin import admin_router

    app.include_router(ingest_router)
    app.include_router(search_router)
    app.include_router(documents_router)
    app.include_router(admin_router)

