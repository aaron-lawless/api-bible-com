# Load the spaCy model
import logging
import re
import uuid

from fastapi import logger
import openai
import spacy
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Config
from app.models.database import ConversationSession, DocumentPage
from app.services.search.constants import _DISTILL_MAX_INPUT_CHARS, _HISTORY_WINDOW, _REWRITE_PROMPT, DISTILL_PROMPT


nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)

# Constants

# Biblical book names that must never be lemmatized
_BIBLE_BOOKS: frozenset[str] = frozenset({
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
    "joshua", "judges", "ruth", "ezra", "nehemiah", "esther", "job",
    "psalms", "psalm", "proverbs", "ecclesiastes", "isaiah", "jeremiah",
    "lamentations", "ezekiel", "daniel", "hosea", "joel", "amos",
    "obadiah", "jonah", "micah", "nahum", "habakkuk", "zephaniah",
    "haggai", "zechariah", "malachi", "matthew", "mark", "luke", "john",
    "acts", "romans", "galatians", "ephesians", "philippians", "colossians",
    "titus", "philemon", "hebrews", "james", "jude", "revelation",
    "corinthians", "thessalonians", "timothy", "peter", "kings", "samuel",
    "chronicles",
})

# Verse reference extraction helpers
_VERSE_COLON_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+):(?P<vs>\d+)(?:\s*[--]\s*(?P<ve>\d+))?",
    re.IGNORECASE,
)
_VERSE_SPACE_DASH_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+)\s+(?P<vs>\d+)\s*[--]\s*(?P<ve>\d+)",
    re.IGNORECASE,
)
_VERSE_SPACE_TO_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+)\s+(?P<vs>\d+)\s+to\s+(?P<ve>\d+)",
    re.IGNORECASE,
)

# Normalize a question by lowercasing, removing stop words and punctuation, and lemmatizing.
# This is used for improving cache hit rates and lowering tokens sent to the emedding model
def normalize_question(text: str) -> str:
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lower = token.text.lower()
        if lower in _BIBLE_BOOKS or token.pos_ == "PROPN":
            tokens.append(lower)
        else:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

# Extract verse reference from text, returning (ref, book, chapter, verse_start, verse_end)
# This helps improve the cache quality for verse-based questions
def extract_verse_reference(text: str) -> tuple[str, str, int, int, int | None] | None:
    for pattern in (_VERSE_COLON_RE, _VERSE_SPACE_DASH_RE, _VERSE_SPACE_TO_RE):
        m = pattern.search(text)
        if m:
            book = m.group("book").strip().lower()
            ch = int(m.group("ch"))
            vs = int(m.group("vs"))
            ve_str = m.group("ve")
            ve = int(ve_str) if ve_str else None
            ref = f"{book} {ch}:{vs}"
            if ve:
                ref += f"-{ve}"
            return ref, book, ch, vs, ve
    return None

# Extract text for a given document and page range, joining pages with a separator
def _extract_page_text(
    document_id: uuid.UUID,
    start_page: int,
    end_page: int,
    db: Session,
) -> str:
    rows = (
        db.execute(
            select(DocumentPage.raw_text)
            .where(
                DocumentPage.document_id == document_id,
                DocumentPage.page_number >= start_page,
                DocumentPage.page_number <= end_page,
            )
            .order_by(DocumentPage.page_number)
        )
        .scalars()
        .all()
    )
    return "\n\n---\n\n".join(rows)

# Gets the information history for the session, this helps with question rewriting (if needed)
def _load_history(session_id: str, db: Session) -> list[dict]:
    """Return the last _HISTORY_WINDOW turn-pairs as a flat [{role, content}] list."""
    session = db.get(ConversationSession, session_id)
    if not session or not session.messages:
        return []
    # Only bringing in the last _HISTORY_WINDOW user+assistant pairs to limit tokens, 
    # we don't want to overwhelm the LLM with too much history for cost and relevance reasons.
    return list(session.messages[-(_HISTORY_WINDOW * 2):])

# Rewrites the question with conversational context
def _rewrite_with_history(
    query: str,
    history: list[dict],
    client: openai.OpenAI,
) -> tuple[str, dict]:
    """Rewrite a follow-up question into a standalone question using conversation history.

    Returns (rewritten_query, token_info).
    """
    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history
    )
    response = client.chat.completions.create(
        model=Config.COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": _REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest question: {query}"
                ),
            },
        ],
        temperature=0,
        max_tokens=256,
    )
    rewritten = response.choices[0].message.content.strip()
    usage = response.usage
    token_info = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return rewritten or query, token_info


def _distill_source(
    query: str,
    doc_title: str,
    section_title: str,
    raw_text: str,
    client: openai.OpenAI,
) -> tuple[str, dict[str, int]]:
    if len(raw_text) > _DISTILL_MAX_INPUT_CHARS:
        logger.warning(
            "Distill input for '%s' truncated from %d to %d chars — add a TOC to avoid this",
            doc_title, len(raw_text), _DISTILL_MAX_INPUT_CHARS,
        )
        raw_text = raw_text[:_DISTILL_MAX_INPUT_CHARS]
    user_msg = (
        f"Query: {query}\n\n"
        f"Source: {doc_title} -- {section_title}\n\n"
        f"Excerpt:\n{raw_text}"
    )
    response = client.chat.completions.create(
        model=Config.COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": DISTILL_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=1024,
    )
    usage = response.usage
    token_info = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return response.choices[0].message.content, token_info
