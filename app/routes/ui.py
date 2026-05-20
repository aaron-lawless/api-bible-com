from fastapi import APIRouter, Path, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

ui_router = APIRouter()

@ui_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def search_ui(request: Request):
    return _templates.TemplateResponse("index.html", {"request": request})