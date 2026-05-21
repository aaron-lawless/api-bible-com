import logging
from datetime import date
from typing import Optional

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config.config import Config
from db.database import get_db
from app.models.database import Document, DocumentPage, DocumentStructure
from app.services.llm.embedder import embed_text
from app.services.ingestion.extractor import build_toc_from_pages, extract_pages
from app.services.ingestion.scraper import scrape_url

logger = logging.getLogger(__name__)

ingest_router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# Target characters per page for URL ingests (~1,000 tokens at ~4 chars/token)
_URL_CHUNK_CHARS = 4_000


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size characters,
    breaking at paragraph boundaries where possible."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _parse_date(date_published: Optional[str]) -> Optional[date]:
    if not date_published:
        return None
    try:
        return date.fromisoformat(date_published)
    except ValueError:
        raise HTTPException(status_code=400, detail="date_published must be YYYY-MM-DD")


@ingest_router.post("/ingest/pdf", status_code=201)
def ingest_pdf(
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    date_published: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    focus_area: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    summary = (summary or "").strip()
    if not summary:
        raise HTTPException(status_code=400, detail="summary is required")

    if not _allowed_file(file.filename):
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT."
        )

    data = file.file.read()
    try:
        pages = extract_pages(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected extraction error for file '%s'", file.filename)
        raise HTTPException(status_code=400, detail="Failed to extract text from the uploaded file") from exc

    if not pages:
        raise HTTPException(status_code=400, detail="No pages could be extracted from the file")

    try:
        summary_embedding = embed_text(summary, api_key=Config.OPENAI_API_KEY)
    except openai.OpenAIError as exc:
        logger.error("OpenAI embedding error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    parsed_date = _parse_date(date_published)

    try:
        doc = Document(
            title=title,
            author=author,
            date_published=parsed_date,
            source=None,
            summary=summary,
            total_pages=len(pages),
            focus_area=focus_area or None,
            summary_embedding=summary_embedding,
        )
        db.add(doc)
        db.flush()

        db.add_all(
            DocumentPage(
                document_id=doc.document_id,
                page_number=page_num,
                raw_text=text,
            )
            for page_num, text in pages
        )

        # Creating the TOC automatically from markdown headings in the extracted text
        toc = build_toc_from_pages(pages)
        if toc:
            db.add_all(
                DocumentStructure(
                    document_id=doc.document_id,
                    section_title=entry["section_title"],
                    start_page=entry["start_page"],
                    end_page=entry["end_page"],
                    level=entry["level"],
                )
                for entry in toc
            )
        else:
            logger.warning("No headings found in PDF '%s'; adding single section for full text", title)
            db.add(DocumentStructure(
                document_id=doc.document_id,
                section_title="Full Document Text",
                start_page=1,
                end_page=len(pages),
                level=1,
            ))

        db.commit()

        logger.info(
            "Ingested PDF '%s' with %d pages (id=%s)", title, len(pages), doc.document_id
        )
        return {
            "document_id": str(doc.document_id),
            "total_pages": len(pages),
            "title": title,
        }
    except Exception as exc:
        db.rollback()
        logger.exception("Error during PDF ingest")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")


@ingest_router.post("/ingest/url", status_code=201)
def ingest_url(
    url: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    date_published: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    focus_area: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    url = (url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    title = (title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")

    summary = (summary or "").strip()
    if not summary:
        raise HTTPException(status_code=400, detail="summary is required")

    try:
        text = scrape_url(url, timeout=Config.SCRAPER_TIMEOUT)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        summary_embedding = embed_text(summary, api_key=Config.OPENAI_API_KEY)
    except openai.OpenAIError as exc:
        logger.error("OpenAI embedding error: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}")

    parsed_date = _parse_date(date_published)
    chunks = _chunk_text(text, _URL_CHUNK_CHARS)

    try:
        doc = Document(
            title=title,
            author=author,
            date_published=parsed_date,
            source=url,
            summary=summary,
            total_pages=len(chunks),
            focus_area=focus_area or None,
            summary_embedding=summary_embedding,
        )
        db.add(doc)
        db.flush()

        # formatting for TOC creation - list of (page_num, text_chunk) tuples
        pages = [(i + 1, chunk) for i, chunk in enumerate(chunks)]

        db.add_all(
            DocumentPage(
                document_id=doc.document_id,
                page_number=page_num,
                raw_text=chunk,
            )
            for page_num, chunk in pages
        )

        #TODO make this a function that can be shared with the PDF ingest route, since it also creates a TOC from markdown headings in the extracted text
        # Creating the TOC automatically from markdown headings in the extracted text
        toc = build_toc_from_pages(pages)
        if toc:
            db.add_all(
                DocumentStructure(
                    document_id=doc.document_id,
                    section_title=entry["section_title"],
                    start_page=entry["start_page"],
                    end_page=entry["end_page"],
                    level=entry["level"],
                )
                for entry in toc
            )
        else:
            logger.warning("No headings found in URL '%s'; adding single section for full text", url)
            db.add(DocumentStructure(
                document_id=doc.document_id,
                section_title="Full Article Text",
                start_page=1,
                end_page=len(chunks),
                level=1,
            ))

        db.commit()

        logger.info(
            "Ingested URL '%s' as document '%s' with %d pages (id=%s)",
            url, title, len(chunks), doc.document_id,
        )
        return {
            "document_id": str(doc.document_id),
            "title": title,
            "total_pages": len(chunks),
        }
    except Exception as exc:
        db.rollback()
        logger.exception("Error during URL ingest")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion")

