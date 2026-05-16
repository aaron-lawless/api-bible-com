import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document, DocumentStructure

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


@documents_router.delete("/documents/{doc_id}")
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    logger.info("Deleted document id=%s", doc_id)
    return {"message": "Document deleted"}


# ?? TOC / Structure endpoints ??????????????????????????????????????????????

class StructureIn(BaseModel):
    section_title: str
    start_page: int
    end_page: int
    level: int = 1

    @model_validator(mode="after")
    def validate_page_range(self):
        if self.start_page < 1:
            raise ValueError("start_page must be >= 1")
        if self.end_page < self.start_page:
            raise ValueError("end_page must be >= start_page")
        return self


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

