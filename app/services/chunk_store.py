import math
import uuid
from dataclasses import dataclass

from sqlalchemy import select, delete, func
from sqlalchemy.orm import Session

from app.database import engine
from app.models import Document, DocumentChunk


@dataclass
class ChunkSearchResult:
    document_id: str
    title: str
    author: str | None
    chunk_index: int
    content: str
    score: float


def store_chunks(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    db: Session,
) -> None:
    chunk_rows = [
        DocumentChunk(
            document_id=uuid.UUID(document_id),
            chunk_index=chunk["chunk_index"],
            content=chunk["content"],
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    db.add_all(chunk_rows)


def count_chunks(document_id: str, db: Session) -> int:
    return db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == uuid.UUID(document_id))
    ) or 0


def delete_chunks(document_id: str, db: Session) -> None:
    db.execute(
        delete(DocumentChunk).where(DocumentChunk.document_id == uuid.UUID(document_id))
    )


def search_chunks(
    query_vector: list[float],
    top_k: int = 10,
    document_ids: list[str] | None = None,
    db: Session = None,
) -> list[ChunkSearchResult]:
    # SQLite doesn't support the pgvector <=> operator — fall back to Python cosine.
    if engine.dialect.name == "sqlite":
        return _search_chunks_sqlite(query_vector, top_k, document_ids, db)
    return _search_chunks_pgvector(query_vector, top_k, document_ids, db)


def _search_chunks_pgvector(
    query_vector: list[float],
    top_k: int,
    document_ids: list[str] | None,
    db: Session,
) -> list[ChunkSearchResult]:
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(DocumentChunk, Document, distance)
        .join(Document, Document.document_id == DocumentChunk.document_id)
    )

    if document_ids:
        stmt = stmt.where(
            DocumentChunk.document_id.in_([uuid.UUID(d) for d in document_ids])
        )

    rows = db.execute(stmt.order_by(distance).limit(top_k)).all()
    return [
        ChunkSearchResult(
            document_id=str(chunk.document_id),
            title=document.title,
            author=document.author,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=1 - float(distance_value),
        )
        for chunk, document, distance_value in rows
    ]


# -- Local SQLite versions of the search functions

def _search_chunks_sqlite(
    query_vector: list[float],
    top_k: int,
    document_ids: list[str] | None,
    db: Session,
) -> list[ChunkSearchResult]:
    stmt = select(DocumentChunk, Document).join(
        Document, Document.document_id == DocumentChunk.document_id
    )
    if document_ids:
        stmt = stmt.where(
            DocumentChunk.document_id.in_([uuid.UUID(d) for d in document_ids])
        )

    rows = db.execute(stmt).all()

    scored = []
    for chunk, document in rows:
        # embedding may be stored as a list or a string representation
        emb = chunk.embedding
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",")]
        else:
            emb = [float(x) for x in emb]
        dist = _cosine_distance(query_vector, emb)
        scored.append((chunk, document, dist))

    scored.sort(key=lambda t: t[2])
    return [
        ChunkSearchResult(
            document_id=str(chunk.document_id),
            title=document.title,
            author=document.author,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=1 - dist,
        )
        for chunk, document, dist in scored[:top_k]
    ]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine distance (1 - cosine_similarity) for SQLite fallback."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)