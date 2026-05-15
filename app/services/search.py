import hashlib
import logging
import re
import uuid

import openai
import spacy
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import QueryCache
from app.services.openai_client import create_openai_client
from app.services import chunk_store
from app.config import Config

logger = logging.getLogger(__name__)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

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

#TODO: we will want to add configuration to point pastors if too deep
#TODO: we will want to shorting of answers to reduce token usage
SYSTEM_PROMPT = """You are a Christian theological assistant helping users understand the Bible through the provided sources.

Answer questions using ONLY the document excerpts provided. Do not draw on outside knowledge.
- Cite the source title without the chunk index for every claim
- Present differing interpretations fairly if they appear in the source material

When the question refers to a specific verse or verse range:
- If the provided excerpts directly address that verse or range, answer from those excerpts
- If the excerpts do not contain enough detail about that specific verse but do cover the surrounding chapter or passage, provide an answer based on that broader context and begin your response with a brief note such as: "I couldn't find specific commentary on that verse, but here is what the sources say about the surrounding passage:"
- If the excerpts contain no relevant information at all, say so clearly — do not speculate or fill in gaps

Format your response using Markdown: use **bold** for emphasis, headings (##, ###) to organise longer answers, and bullet lists where appropriate.
Write with clarity and pastoral warmth, grounded entirely in the provided documents."""

# normalizing the question to reduce noise in the vector search & token usage
# in additional this is helpful for exact match searches of previous questions/answers to avoid extra AI API calls
def normalize_question(text: str) -> str:
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lower = token.text.lower()
        # Preserve biblical book names and proper nouns without lemmatization
        if lower in _BIBLE_BOOKS or token.pos_ == "PROPN":
            tokens.append(lower)
        else:
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Verse reference extraction & canonicalisation
# ---------------------------------------------------------------------------
# Capture groups for each pattern:
#   (book) (chapter) : (verse_start) [- (verse_end)]
#
# Pattern A:  "Romans 8:1"  /  "Romans 8:1-3"  /  "Judges 8:1–6"
_VERSE_COLON_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+):(?P<vs>\d+)(?:\s*[-–]\s*(?P<ve>\d+))?",
    re.IGNORECASE,
)
# Pattern B:  "Romans 8 1-3"  /  "Romans 8 1–3"
_VERSE_SPACE_DASH_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+)\s+(?P<vs>\d+)\s*[-–]\s*(?P<ve>\d+)",
    re.IGNORECASE,
)
# Pattern C:  "Romans 8 1 to 3"
_VERSE_SPACE_TO_RE = re.compile(
    r"(?P<book>[1-3]?\s*[a-zA-Z]+)\s+(?P<ch>\d+)\s+(?P<vs>\d+)\s+to\s+(?P<ve>\d+)",
    re.IGNORECASE,
)


def extract_verse_reference(text: str) -> str | None:
    """Return a canonical verse reference string if one is found in *text*.

    Canonical form: "<book> <chapter>:<verse_start>[-<verse_end>]"
    All lowercase, e.g. "romans 8:1-3", "judges 8:1".  Returns None when no
    verse reference is detected.
    """
    for pattern in (_VERSE_COLON_RE, _VERSE_SPACE_DASH_RE, _VERSE_SPACE_TO_RE):
        m = pattern.search(text)
        if m:
            book = m.group("book").strip().lower()
            ch = m.group("ch")
            vs = m.group("vs")
            ve = m.group("ve")  # may be None
            ref = f"{book} {ch}:{vs}"
            if ve:
                ref += f"-{ve}"
            return ref
    return None


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


def answer_question(
    query: str,
    top_k: int = 10,
    document_ids: list[str] | None = None,
    api_key: str | None = None,
    db: Session | None = None,
    session_id: str | None = None,
) -> dict:
    client = create_openai_client(api_key)
    try:
        # Normalize the question for better search and potential caching
        normalized_query = normalize_question(query)
        question_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        verse_ref = extract_verse_reference(query)

        # Step 1: Exact match search for previously answered questions to avoid unnecessary API calls

        stmt = (
            select(QueryCache)
            .where(QueryCache.question_hash == question_hash, QueryCache.cache_hit == False)  # noqa: E712
            .limit(1)
        )
        row = db.execute(stmt).scalar_one_or_none()
        if row:
            logger.info("Cache hit for question: %s", query[:80])
            _insert_cache_row(
                db=db,
                question_raw=query,
                question_normalized=normalized_query,
                question_hash=question_hash,
                embedding=row.embedding,
                response=row.response,
                cache_hit=True,
                sources=row.sources,
                cache_hit_type="exact",
                cache_source_id=row.query_id,
                session_id=session_id or "",
                verse_reference=verse_ref,
            )
            return {"answer": row.response, "sources": row.sources or []}

        # Step 2: Vector match from previous searches
        # For verse-specific questions we restrict candidates to rows with the
        # same canonical verse reference, so "Romans 8:1-2" never matches
        # "Romans 8:1-6" even though their embeddings are very similar.

        # creating the embedding for the question to perform vector search
        embed_response = client.embeddings.create(
            model=Config.EMBEDDING_MODEL,
            input=[normalized_query],
        )

        question_embedding = embed_response.data[0].embedding

        distance_expr = QueryCache.embedding.cosine_distance(question_embedding).label("distance")
        vector_stmt = (
            select(QueryCache, distance_expr)
            .where(QueryCache.cache_hit == False)  # noqa: E712
            .order_by(distance_expr)
            .limit(1)
        )
        if verse_ref is not None:
            # Only match against rows that reference the exact same verse(s)
            vector_stmt = vector_stmt.where(QueryCache.verse_reference == verse_ref)
        else:
            # Exclude verse-specific rows from general vector matching
            vector_stmt = vector_stmt.where(QueryCache.verse_reference.is_(None))

        row = db.execute(vector_stmt).first()

        if row:
            cached, distance = row
            similarity = 1 - float(distance)
            # TODO: we will want to have this as a configurable threshold
            print(f"Vector cache similarity: {similarity:.4f} for question: {query[:80]}")
            if similarity > 0.85:
                _insert_cache_row(
                    db=db,
                    question_raw=query,
                    question_normalized=normalized_query,
                    question_hash=question_hash,
                    embedding=question_embedding,
                    response=cached.response,
                    cache_hit=True,
                    sources=cached.sources,
                    cache_hit_type="vector",
                    cache_source_id=cached.query_id,
                    similarity_score=similarity,
                    token_information={
                        "embedding_tokens": embed_response.usage.prompt_tokens,
                    },
                    session_id=session_id or "",
                    verse_reference=verse_ref,
                )
                return {"answer": cached.response, "sources": cached.sources or []}

        # Step 3: Full AI generation (RAG & embedding search)

        hits = chunk_store.search_chunks(
            query_vector=question_embedding,
            top_k=top_k,
            document_ids=document_ids,
            db=db,
        )

        seen_document_ids: set[str] = set()
        sources = []
        context_parts = []
        for hit in hits:
            doc_id = str(hit.document_id)
            if doc_id not in seen_document_ids:
                seen_document_ids.add(doc_id)
                sources.append(
                    {
                        "document_id": doc_id,
                        "title": hit.title,
                        "author": hit.author,
                        "source": hit.source
                    }
                )
            context_parts.append(
                f"[Source: {hit.title}, chunk {hit.chunk_index}]\n{hit.content}"
            )

        context = "\n\n---\n\n".join(context_parts)
        user_message = f"Question: {query}\n\nDocument excerpts:\n\n{context}"

        logger.info("Generating answer for query: %s", query[:80])

        completion = client.chat.completions.create(
            model=Config.COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0,
        )

        answer = completion.choices[0].message.content

        _insert_cache_row(
            db=db,
            question_raw=query,
            question_normalized=normalized_query,
            question_hash=question_hash,
            embedding=question_embedding,
            response=answer,
            cache_hit=False,
            sources=sources,
            token_information={
                "embedding_tokens": embed_response.usage.prompt_tokens,
                "completion_tokens": completion.usage.total_tokens,
            },
            session_id=session_id or "",
            verse_reference=verse_ref,
        )

        return {"answer": answer, "sources": sources}
    finally:
        client.close()
