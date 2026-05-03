import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Uuid, Text, Date, DateTime, Integer, ForeignKey, UniqueConstraint, Boolean, JSON
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    author = Column(Text)
    isbn = Column(Text)
    date_published = Column(Date)
    description = Column(Text)
    source = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self):
        return {
            "document_id": str(self.document_id),
            "title": self.title,
            "author": self.author,
            "isbn": self.isbn,
            "date_published": (
                self.date_published.isoformat() if self.date_published else None
            ),
            "description": self.description,
            "source": self.source,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    chunk_id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )


class QueryCache(Base):
    __tablename__ = "query_cache"

    query_id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_raw = Column(Text, nullable=False)
    question_normalized = Column(Text, nullable=False)
    question_hash = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    response = Column(Text)
    cache_hit = Column(Boolean, nullable=False)
    cache_hit_type = Column(Text)
    token_information = Column(JSON)
    similarity_score = Column(Text)
    session_id = Column(Text, nullable=False)
    cache_source_id = Column(Uuid(as_uuid=True)) # This will reference other querycache rows
    sources = Column(JSON)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

