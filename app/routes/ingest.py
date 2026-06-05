import logging
import uuid as uuid_mod
from datetime import date
from typing import Optional

import openai
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config.config import Config
from db.database import SessionLocal, get_db
from app.models.database import Document, DocumentPage, DocumentStructure, IngestJob
from app.services.llm.embedder import embed_text
from app.services.ingestion.extractor import build_toc_from_ai, build_toc_from_pages, extract_pages
from app.services.ingestion.scraper import scrape_url

logger = logging.getLogger(__name__)

ingest_router = APIRouter()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# Target characters per page for URL ingests (~1,000 tokens at ~4 chars/token)
_URL_CHUNK_CHARS = 4_000

# TODO: Another util function, move somewhere common
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

# TODO: Another util function, move somewhere
def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# TODO: This is currently duplicated from pyrocker.config move this somwhere common
def _parse_date(date_published: Optional[str]) -> Optional[date]:
    if not date_published:
        return None
    try:
        return date.fromisoformat(date_published)
    except ValueError:
        raise HTTPException(status_code=400, detail="date_published must be YYYY-MM-DD")

# TODO: This should be somewhere else, not sure where
def _update_job(job_id: uuid_mod.UUID, status: str, document_id=None, error: str = None) -> None:
    """Update IngestJob status in a fresh DB session (safe to call from background threads)."""
    db = SessionLocal()
    try:
        job = db.get(IngestJob, job_id)
        if job:
            job.status = status
            if document_id is not None:
                job.document_id = document_id
            if error is not None:
                job.error = error
            db.commit()
    except Exception:
        logger.exception("Failed to update ingest job %s to status '%s'", job_id, status)
    finally:
        db.close()

# TODO: feel like this should be somewhere else
def _build_and_save_toc(
    db: Session,
    doc: Document,
    pages: list[tuple[int, str]],
    use_ai_toc: bool,
    label: str,
) -> None:
    if use_ai_toc:
        toc = build_toc_from_ai(pages, doc.title, Config.OPENAI_API_KEY, Config.COMPLETION_MODEL)
    else:
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
        logger.warning("No headings found for '%s'; adding single section for full text", label)
        db.add(DocumentStructure(
            document_id=doc.document_id,
            section_title="Full Document Text",
            start_page=1,
            end_page=len(pages),
            level=1,
        ))

#TODO: Could we generalize the _run_pdf_ingest_job and _run_url_ingest_job functions to reduce code duplication, since they share a lot of logic around creating the Document, adding pages, building TOC, and error handling? The main differences are in how the text is obtained (PDF extraction vs URL scraping) and how it's chunked (pages vs fixed-size chunks).

def _run_pdf_ingest_job(
    job_id: uuid_mod.UUID,
    title: str,
    author: Optional[str],
    parsed_date: Optional[date],
    summary: str,
    focus_area: Optional[str],
    pages: list[tuple[int, str]],
    use_ai_toc: bool,
) -> None:
    _update_job(job_id, "running")
    db = SessionLocal()
    try:
        try:
            summary_embedding = embed_text(summary, api_key=Config.OPENAI_API_KEY)
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI embedding error: {exc}") from exc

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

        _build_and_save_toc(db, doc, pages, use_ai_toc, title)
        db.commit()

        logger.info("PDF ingest job %s done: '%s' (%d pages, id=%s)", job_id, title, len(pages), doc.document_id)
        _update_job(job_id, "done", document_id=doc.document_id)

    except Exception as exc:
        db.rollback()
        logger.exception("PDF ingest job %s failed for '%s'", job_id, title)
        _update_job(job_id, "failed", error=str(exc))
    finally:
        db.close()

def _run_url_ingest_job(
    job_id: uuid_mod.UUID,
    url: str,
    title: str,
    author: Optional[str],
    parsed_date: Optional[date],
    summary: str,
    focus_area: Optional[str],
    chunks: list[str],
    use_ai_toc: bool,
) -> None:
    _update_job(job_id, "running")
    db = SessionLocal()
    try:
        try:
            summary_embedding = embed_text(summary, api_key=Config.OPENAI_API_KEY)
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI embedding error: {exc}") from exc

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

        pages = [(i + 1, chunk) for i, chunk in enumerate(chunks)]
        db.add_all(
            DocumentPage(
                document_id=doc.document_id,
                page_number=page_num,
                raw_text=chunk,
            )
            for page_num, chunk in pages
        )

        _build_and_save_toc(db, doc, pages, use_ai_toc, url)
        db.commit()

        logger.info("URL ingest job %s done: '%s' (%d chunks, id=%s)", job_id, title, len(chunks), doc.document_id)
        _update_job(job_id, "done", document_id=doc.document_id)

    except Exception as exc:
        db.rollback()
        logger.exception("URL ingest job %s failed for '%s'", job_id, title)
        _update_job(job_id, "failed", error=str(exc))
    finally:
        db.close()

# Endpoint to ingest a PDF file with metadata and optional AI-generated TOC, processed in the background
@ingest_router.post("/ingest/pdf", status_code=202)
def ingest_pdf(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    date_published: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    focus_area: Optional[str] = Form(None),
    ai_toc: Optional[str] = Form(None),
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

    parsed_date = _parse_date(date_published)

    job = IngestJob(title=title, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        _run_pdf_ingest_job,
        job.job_id,
        title,
        author or None,
        parsed_date,
        summary,
        focus_area or None,
        pages,
        ai_toc == "true",
    )

    logger.info("PDF ingest job %s queued for '%s' (%d pages)", job.job_id, title, len(pages))
    return {"job_id": str(job.job_id), "status": "pending", "title": title, "total_pages": len(pages)}

# Endpoint to ingest content from a URL, with metadata and optional AI-generated TOC, processed in the background
@ingest_router.post("/ingest/url", status_code=202)
def ingest_url(
    background_tasks: BackgroundTasks,
    url: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    date_published: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    focus_area: Optional[str] = Form(None),
    ai_toc: Optional[str] = Form(None),
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

    parsed_date = _parse_date(date_published)
    chunks = _chunk_text(text, _URL_CHUNK_CHARS)

    job = IngestJob(title=title, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(
        _run_url_ingest_job,
        job.job_id,
        url,
        title,
        author or None,
        parsed_date,
        summary,
        focus_area or None,
        chunks,
        ai_toc == "true",
    )

    logger.info("URL ingest job %s queued for '%s'", job.job_id, title)
    return {"job_id": str(job.job_id), "status": "pending", "title": title}

# Get ingest job status and details (for polling from frontend)
@ingest_router.get("/ingest/jobs/{job_id}")
def get_ingest_job(job_id: uuid_mod.UUID, db: Session = Depends(get_db)):
    job = db.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return job.to_dict()

