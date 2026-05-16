import logging
import uuid
from pathlib import Path
from typing import Optional, List

import openai
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.config import Config
from app.database import get_db
from app.models import QueryCache
from app.services.search import answer_question, answer_question_stream

_templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

logger = logging.getLogger(__name__)

search_router = APIRouter()


@search_router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def search_ui(request: Request):
    return _templates.TemplateResponse("index.html", {"request": request})


class SearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query is required")
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
def search(request: Request, body: SearchRequest, db: Session = Depends(get_db)):
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    session_id = request.session["session_id"]

    try:
        result = answer_question(
            query=body.query,
            document_ids=body.document_ids,
            api_key=Config.OPENAI_API_KEY,
            db=db,
            session_id=session_id,
        )
        logger.info("Search completed for query: %s", body.query[:80])
        return result
    except openai.OpenAIError as exc:
        logger.error("OpenAI error during search: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")
    except Exception as exc:
        logger.error("Unexpected error during search: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


def _parse_document_ids(raw: Optional[List[str]]) -> Optional[List[str]]:
    if raw is None:
        return None
    validated = []
    for item in raw:
        try:
            validated.append(str(uuid.UUID(item)))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid document_id: {item}")
    return validated


@search_router.get("/search/stream")
async def search_stream(
    request: Request,
    query: str = Query(..., min_length=1),
    document_ids: Optional[List[str]] = Query(None),
    db: Session = Depends(get_db),
):
    """Server-Sent Events endpoint. Streams pipeline thinking steps then the final answer."""
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    session_id = request.session["session_id"]

    validated_ids = _parse_document_ids(document_ids)

    return EventSourceResponse(
        answer_question_stream(
            query=query,
            document_ids=validated_ids,
            api_key=Config.OPENAI_API_KEY,
            db=db,
            session_id=session_id,
        )
    )


@search_router.get("/questions")
def list_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return a paginated list of previously answered unique questions."""
    offset = (page - 1) * page_size

    base_filter = [QueryCache.cache_hit == False]  # noqa: E712
    if search:
        base_filter.append(QueryCache.question_raw.ilike(f"%{search}%"))

    total = db.execute(
        select(func.count()).select_from(QueryCache).where(*base_filter)
    ).scalar_one()

    rows = db.execute(
        select(QueryCache.query_id, QueryCache.question_raw, QueryCache.sources)
        .where(*base_filter)
        .order_by(QueryCache.created_at.desc())
        .offset(offset)
        .limit(page_size)
    ).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "questions": [
            {
                "query_id": str(row.query_id),
                "question": row.question_raw,
                "sources": row.sources or [],
            }
            for row in rows
        ],
    }


@search_router.get("/questions/{query_id}")
def get_question_answer(query_id: str, db: Session = Depends(get_db)):
    """Retrieve a previously answered question by ID. No API call is made."""
    try:
        uid = uuid.UUID(query_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid query_id")

    row = db.execute(
        select(QueryCache).where(QueryCache.query_id == uid, QueryCache.cache_hit == False)  # noqa: E712
    ).scalar_one_or_none()

    if not row:
        raise HTTPException(status_code=404, detail="Question not found")

    return {
        "query_id": str(row.query_id),
        "question": row.question_raw,
        "answer": row.response,
        "sources": row.sources or [],
    }


_templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

logger = logging.getLogger(__name__)

