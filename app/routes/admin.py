import hmac
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from config.config import Config

logger = logging.getLogger(__name__)

admin_router = APIRouter(prefix="/admin", include_in_schema=False)

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Setting the admin password from environment configuration.
ADMIN_PASSWORD = Config.ADMIN_PASSWORD


@admin_router.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    if request.session.get("admin_authenticated"):
        return RedirectResponse("/admin/", status_code=302)
    return templates.TemplateResponse(
        "admin_login.html", {"request": request, "error": None}
    )


@admin_router.post("/login", response_class=HTMLResponse)
def login_post(request: Request, password: str = Form(...)):
    if password == ADMIN_PASSWORD:
        request.session["admin_authenticated"] = True
        logger.info("Admin login successful")
        return RedirectResponse("/admin/", status_code=302)
    logger.warning("Failed admin login attempt")
    return templates.TemplateResponse(
        "admin_login.html",
        {"request": request, "error": "Invalid password."},
        status_code=200,
    )


@admin_router.get("/logout")
def logout(request: Request):
    request.session.pop("admin_authenticated", None)
    return RedirectResponse("/admin/login", status_code=302)


@admin_router.get("", response_class=HTMLResponse)
def dashboard(request: Request):
    if not request.session.get("admin_authenticated"):
        return RedirectResponse("/admin/login", status_code=302)
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})

