import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.services import chunk_store

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
    data["chunk_count"] = chunk_store.count_chunks(str(doc_id), db)
    return data


@documents_router.delete("/documents/{doc_id}")
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_store.delete_chunks(str(doc_id), db)

    db.delete(doc)
    db.commit()
    logger.info("Deleted document id=%s", doc_id)
    return {"message": "Document deleted"}

