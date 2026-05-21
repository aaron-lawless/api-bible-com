import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.database import get_db
from app.models.database import Document, DocumentPage, DocumentStructure
from app.models.request import StructureIn

logger = logging.getLogger(__name__)

documents_router = APIRouter()


@documents_router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = (
        db.execute(select(Document).order_by(Document.created_at.desc()))
        .scalars()
        .all()
    )
    return [doc.to_dict() for doc in docs]


@documents_router.get("/documents/{doc_id}")
def get_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    data = doc.to_dict()
    data["structure_count"] = len(doc.structures)
    return data


@documents_router.get("/documents/{doc_id}/pages")
def list_document_pages(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = (
        db.execute(
            select(DocumentPage.page_number)
            .where(DocumentPage.document_id == doc_id)
            .order_by(DocumentPage.page_number)
        )
        .all()
    )
    return {
        "document_id": str(doc_id),
        "total_pages": doc.total_pages,
        "pages": [r.page_number for r in rows],
    }


@documents_router.get("/documents/{doc_id}/pages/{page_number}")
def get_document_page(
    doc_id: uuid.UUID,
    page_number: int,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")

    page = (
        db.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id == doc_id,
                DocumentPage.page_number == page_number,
            )
            .limit(1)
        )
        .scalar_one_or_none()
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Page not found")

    return {
        "document_id": str(doc_id),
        "title": doc.title,
        "page_number": page.page_number,
        "total_pages": doc.total_pages,
        "raw_text": page.raw_text,
    }


@documents_router.delete("/documents/{doc_id}")
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    logger.info("Deleted document id=%s", doc_id)
    return {"message": "Document deleted"}

@documents_router.get("/documents/{doc_id}/structures")
def list_structures(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = (
        db.execute(
            select(DocumentStructure)
            .where(DocumentStructure.document_id == doc_id)
            .order_by(DocumentStructure.start_page)
        )
        .scalars()
        .all()
    )
    return [r.to_dict() for r in rows]


@documents_router.post("/documents/{doc_id}/structures", status_code=201)
def create_structure(
    doc_id: uuid.UUID, body: StructureIn, db: Session = Depends(get_db)
):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    if body.end_page > doc.total_pages:
        raise HTTPException(
            status_code=400,
            detail=f"end_page ({body.end_page}) exceeds document total_pages ({doc.total_pages})",
        )

    entry = DocumentStructure(
        document_id=doc_id,
        section_title=body.section_title,
        start_page=body.start_page,
        end_page=body.end_page,
        level=body.level,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info("Created structure entry '%s' for document id=%s", body.section_title, doc_id)
    return entry.to_dict()


@documents_router.put("/documents/{doc_id}/structures/{struct_id}")
def update_structure(
    doc_id: uuid.UUID,
    struct_id: uuid.UUID,
    body: StructureIn,
    db: Session = Depends(get_db),
):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    entry = db.get(DocumentStructure, struct_id)
    if entry is None or entry.document_id != doc_id:
        raise HTTPException(status_code=404, detail="Structure entry not found")

    if body.end_page > doc.total_pages:
        raise HTTPException(
            status_code=400,
            detail=f"end_page ({body.end_page}) exceeds document total_pages ({doc.total_pages})",
        )

    entry.section_title = body.section_title
    entry.start_page = body.start_page
    entry.end_page = body.end_page
    entry.level = body.level
    db.commit()
    db.refresh(entry)
    return entry.to_dict()


@documents_router.delete("/documents/{doc_id}/structures/{struct_id}")
def delete_structure(
    doc_id: uuid.UUID, struct_id: uuid.UUID, db: Session = Depends(get_db)
):
    entry = db.get(DocumentStructure, struct_id)
    if entry is None or entry.document_id != doc_id:
        raise HTTPException(status_code=404, detail="Structure entry not found")

    db.delete(entry)
    db.commit()
    return {"message": "Structure entry deleted"}

