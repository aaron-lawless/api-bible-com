from datetime import datetime, timezone
import uuid

from sqlalchemy.orm import Session
from app.models.database import ConversationSession, QueryCache

# Used for caching search results.
def _insert_cache_row(
    db: Session,
    *,
    question_raw: str,
    question_normalized: str,
    question_hash: str,
    embedding: list[float],
    response: str,
    cache_hit: bool,
    sources: list[dict] | None = None,
    cache_hit_type: str | None = None,
    cache_source_id: uuid.UUID | None = None,
    similarity_score: float | None = None,
    token_information: dict | None = None,
    session_id: str = "",
    verse_reference: str | None = None,
) -> None:
    row = QueryCache(
        question_raw=question_raw,
        question_normalized=question_normalized,
        question_hash=question_hash,
        embedding=embedding,
        response=response,
        cache_hit=cache_hit,
        sources=sources or [],
        cache_hit_type=cache_hit_type,
        cache_source_id=cache_source_id,
        similarity_score=str(similarity_score) if similarity_score is not None else None,
        token_information=token_information or {},
        session_id=session_id,
        verse_reference=verse_reference,
    )
    db.add(row)
    db.commit()

#  This is used to update the conversation history after generating a response
def _append_history(
    session_id: str,
    user_content: str,
    assistant_content: str,
    db: Session,
) -> None:
    """Append a user+assistant turn to the session's conversation history."""
    session = db.get(ConversationSession, session_id)
    if session is None:
        session = ConversationSession(session_id=session_id, messages=[])
        db.add(session)
    messages = list(session.messages or [])
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_content})
    session.messages = messages
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
