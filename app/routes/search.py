import logging
import uuid
from pathlib import Path
from typing import Optional, List

import openai
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.config import Config
from app.database import get_db
from app.services.search import answer_question

_templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

logger = logging.getLogger(__name__)

search_router = APIRouter()


@search_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def search_ui(request: Request):
    return _templates.TemplateResponse("index.html", {"request": request})


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    document_ids: Optional[List[str]] = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query is required")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be a positive integer")
        return v

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            try:
                return [str(uuid.UUID(str(d))) for d in v]
            except ValueError:
                raise ValueError("document_ids contains an invalid UUID")
        return v


@search_router.post("/search")
def search(body: SearchRequest, db: Session = Depends(get_db)):
    try:
        result = answer_question(
            query=body.query,
            top_k=body.top_k,
            document_ids=body.document_ids,
            api_key=Config.OPENAI_API_KEY,
            db=db,
        )
        logger.info("Search completed for query: %s", body.query[:80])
        return result
    except openai.OpenAIError as exc:
        logger.error("OpenAI error during search: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")
    except Exception as exc:
        logger.error("Unexpected error during search: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")

