import hashlib
import logging
import uuid

import openai
import spacy
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import QueryCache
from app.services.openai_client import create_openai_client
from app.services import chunk_store

logger = logging.getLogger(__name__)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

#TODO: we will want to add configuration to point pastors if too deep
#TODO: we will want to query our previous answers to avoid extra API calls if the same question is asked multiple times
COMPLETION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
SYSTEM_PROMPT = """You are a Christian theological assistant helping users understand the Bible through the provided sources.

Answer questions using ONLY the document excerpts provided. Do not draw on outside knowledge.
- Cite the source title without the chunk index for every claim
- If the provided excerpts do not contain enough information to answer, say so clearly — do not speculate or fill in gaps
- Present differing interpretations fairly if they appear in the source material

Format your response using Markdown: use **bold** for emphasis, headings (##, ###) to organise longer answers, and bullet lists where appropriate.
Write with clarity and pastoral warmth, grounded entirely in the provided documents."""

# normalizing the question to reduce noise in the vector search & token usage
# in additional this is helpful for exact match searches of previous questions/answers to avoid extra AI API calls
def normalize_question(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


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
            )
            return {"answer": row.response, "sources": row.sources or []}

        # Step 2: Vector match from previous searches

        # creating the embedding for the question to perform vector search
        embed_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[normalized_query],
        )

        question_embedding = embed_response.data[0].embedding

        distance_expr = QueryCache.embedding.cosine_distance(question_embedding).label("distance")
        stmt = (
            select(QueryCache, distance_expr)
            .where(QueryCache.cache_hit == False)  # only consider the previous questions that were not cache hits themselves
            .order_by(distance_expr)
            .limit(1)
        )
        row = db.execute(stmt).first()

        if row:
            cached, distance = row
            similarity = 1 - float(distance)
            # TODO: we will want to have this as a configurable threshold
            if similarity > 0.92:
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
            model=COMPLETION_MODEL,
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
        )

        return {"answer": answer, "sources": sources}
    finally:
        client.close()
