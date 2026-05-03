import logging
from datetime import date
from typing import Optional

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import Config
from app.database import get_db
from app.models import Document
from app.services import chunk_store
from app.services.chunker import chunk_text
from app.services.embedder import embed_chunks
from app.services.extractor import extract_text

logger = logging.getLogger(__name__)

ingest_router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@ingest_router.post("/ingest", status_code=201)
def ingest(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    isbn: Optional[str] = Form(None),
    date_published: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT."
        )

    data = file.file.read()
    try:
        text = extract_text(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    chunks = chunk_text(text)

    try:
        embeddings = embed_chunks(chunks, api_key=Config.OPENAI_API_KEY)
    except openai.OpenAIError as exc:
        logger.error("OpenAI embedding error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    parsed_date = None
    if date_published:
        try:
            parsed_date = date.fromisoformat(date_published)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="date_published must be YYYY-MM-DD"
            )

    try:
        doc = Document(
            title=title,
            author=author,
            isbn=isbn,
            date_published=parsed_date,
            description=description,
        )
        db.add(doc)
        db.flush()

        chunk_store.store_chunks(str(doc.document_id), chunks, embeddings, db)

        db.commit()

        logger.info(
            "Ingested document '%s' with %d chunks (id=%s)", title, len(chunks), doc.document_id
        )
        return {
            "document_id": str(doc.document_id),
            "chunk_count": len(chunks),
            "title": title,
        }
    except Exception as exc:
        db.rollback()
        logger.error("Error during ingest: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal server error during ingestion"
        )

