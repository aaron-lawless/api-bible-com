import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Uuid, Text, Date, DateTime, Integer, ForeignKey, UniqueConstraint, Boolean, JSON, Index
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    document_id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    author = Column(Text)
    date_published = Column(Date)
    # For web pages: original URL. For PDFs: NULL.
    source = Column(Text)
    summary = Column(Text, nullable=False)
    total_pages = Column(Integer, nullable=False)
    focus_area = Column(Text)
    summary_embedding = Column(Vector(1536))
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    structures = relationship(
        "DocumentStructure",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentStructure.start_page",
    )
    pages = relationship(
        "DocumentPage",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentPage.page_number",
    )

    def to_dict(self):
        return {
            "document_id": str(self.document_id),
            "title": self.title,
            "author": self.author,
            "date_published": (
                self.date_published.isoformat() if self.date_published else None
            ),
            "source": self.source,
            "summary": self.summary,
            "total_pages": self.total_pages,
            "focus_area": self.focus_area,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }


class DocumentStructure(Base):
    __tablename__ = "document_structures"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    section_title = Column(Text, nullable=False)
    start_page = Column(Integer, nullable=False)
    end_page = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False, default=1)

    document = relationship("Document", back_populates="structures")

    __table_args__ = (
        Index("idx_structures_doc_id", "document_id"),
        Index("idx_structures_doc_level", "document_id", "level"),
    )

    def to_dict(self):
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "section_title": self.section_title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
        }


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    page_number = Column(Integer, nullable=False)
    raw_text = Column(Text, nullable=False)

    document = relationship("Document", back_populates="pages")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_page_number"),
        Index("idx_pages_doc_range", "document_id", "page_number"),
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
    verse_reference = Column(Text, nullable=True)  # Canonical form e.g. "romans 8:1-2", NULL for non-verse questions
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ConversationSession(Base):
    """Stores rolling per-session conversation history for follow-up questions."""

    __tablename__ = "conversation_sessions"

    session_id = Column(Text, primary_key=True)
    # Flat list of {"role": "user"|"assistant", "content": str} dicts
    messages = Column(JSON, nullable=False, default=list)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

