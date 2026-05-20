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
from app.models.database import QueryCache
from app.models.request import SearchRequest
from app.services.search import answer_question

logger = logging.getLogger(__name__)

search_router = APIRouter()

@search_router.get("/search")
async def search(
    request: Request,
    query: str = Query(...),
    db: Session = Depends(get_db)
):
    """Server-Sent Events endpoint. Streams pipeline thinking steps then the final answer."""
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    session_id = request.session["session_id"]

    return EventSourceResponse(
        answer_question(
            query=query,
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

