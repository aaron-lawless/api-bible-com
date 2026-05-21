import hashlib
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import AsyncGenerator

import openai
import spacy
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database import Document, DocumentStructure, DocumentPage, QueryCache, ConversationSession
from app.services.llm.openai_client import _embed_query, create_openai_client
from config.config import Config
from app.services.search.cache import _append_history, _insert_cache_row
from app.services.search.constants import (
    _DISTILL_MAX_INPUT_CHARS,
    _DISTILL_THRESHOLD_CHARS,
    _DISTILL_THRESHOLD_PAGES,
    _REWRITE_PROMPT,
    DISTILL_PROMPT,
    NAV_PROMPT,
    ROUTING_PROMPT,
    SYSTEM_PROMPT,
)
from app.services.search.utils import _distill_source, _extract_page_text, _load_history, _rewrite_with_history, extract_verse_reference, normalize_question

logger = logging.getLogger(__name__)

# TODO: Create models that the llm should respond with

# ---------------------------------------------------------------------------
# Tier 1 -- Document Router
# ---------------------------------------------------------------------------

def _tier1_route_documents(
    query: str,
    query_embedding: list[float],
    client: openai.OpenAI,
    db: Session,
) -> list[Document]:
    """Return the list of Document objects to search for this query."""

    # pgvector pre-filter: cosine similarity on summary embeddings
    top_k = Config.TIER1_PREFILTER_TOP_K
    distance_expr = Document.summary_embedding.cosine_distance(query_embedding).label("distance")
    candidates = (
        db.execute(
            select(Document, distance_expr)
            .where(Document.summary_embedding.is_not(None))
            .order_by(distance_expr)
            .limit(top_k)
        )
        .all()
    )

    if not candidates:
        return []

    # Build the LLM classifier prompt
    doc_summaries = "\n\n".join(
        f"[{i+1}] document_id={str(row.Document.document_id)}\n"
        f"Title: {row.Document.title}\n"
        f"Author: {row.Document.author or 'Unknown'}\n"
        f"Focus: {row.Document.focus_area or 'N/A'}\n"
        f"Summary: {row.Document.summary}"
        for i, row in enumerate(candidates)
    )

    routing_prompt = ROUTING_PROMPT.format(
        query=query,
        doc_summaries=doc_summaries,
    )

    response = client.chat.completions.create(
        model=Config.COMPLETION_MODEL,
        messages=[{"role": "user", "content": routing_prompt}],
        temperature=0,
        max_tokens=256,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        selected_ids = json.loads(raw)
        if not isinstance(selected_ids, list):
            raise ValueError("Expected a JSON array")
    except (json.JSONDecodeError, ValueError):
        logger.warning("Tier 1 router returned unexpected output: %s", raw)
        # Fall back to top candidate
        selected_ids = [str(candidates[0].Document.document_id)]

    docs = []
    id_to_doc = {str(row.Document.document_id): row.Document for row in candidates}
    for sid in selected_ids:
        doc = id_to_doc.get(str(sid))
        if doc:
            docs.append(doc)

    return docs


# ---------------------------------------------------------------------------
# Tier 2 -- Section Navigator
# ---------------------------------------------------------------------------

def _tier2_navigate_section(
    query: str,
    doc: Document,
    client: openai.OpenAI,
    db: Session,
) -> tuple[int, int, str]:
    """Return (start_page, end_page, section_title) for the best-matching section."""
    structures = (
        db.execute(
            select(DocumentStructure)
            .where(DocumentStructure.document_id == doc.document_id)
            .order_by(DocumentStructure.start_page)
        )
        .scalars()
        .all()
    )

    # Fallback: no TOC -- return full document
    if not structures:
        logger.info("No TOC for '%s', using full document range", doc.title)
        return 1, doc.total_pages, "Full Document"

    toc_text = "\n".join(
        f"[{i+1}] level={s.level} pages={s.start_page}-{s.end_page}: {s.section_title}"
        for i, s in enumerate(structures)
    )

    nav_prompt = NAV_PROMPT.format(
        doc_title=doc.title,
        query=query,
        toc_text=toc_text,
    )

    response = client.chat.completions.create(
        model=Config.COMPLETION_MODEL,
        messages=[{"role": "user", "content": nav_prompt}],
        temperature=0,
        max_tokens=128,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        idx = int(parsed["index"]) - 1
        if 0 <= idx < len(structures):
            s = structures[idx]
            return s.start_page, s.end_page, s.section_title
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("Tier 2 navigator returned unexpected output: %s", raw)

    # Fallback to first structure entry
    s = structures[0]
    return s.start_page, s.end_page, s.section_title


# ---------------------------------------------------------------------------
# Core pipeline 
# ---------------------------------------------------------------------------

def _run_pipeline(
    query: str,
    api_key: str | None,
    db: Session,
    session_id: str,
    on_thinking: callable,  # fn(message: str) -> None
) -> dict:
    """Pipeline has 3 tiers:
    1. Exact cache match on normalized question hash
    2. Semantic cache match using vector search on question embedding
    3. Agentic pipeline with: document routing (Tier 1), section navigation (Tier 2), page extraction, and synthesis

    *on_thinking* is called synchronously at each stage with a human-readable
    status string, enabling SSE streaming without duplicating pipeline logic.

    Returns {"answer": str, "sources": list[dict]}.
    """
    client = create_openai_client(api_key)
    try:
        # Load conversation history from sessions
        history = _load_history(session_id, db)
        rewrite_tokens: dict | None = None

        # Rewriting the query based on conversation history (if applicable) 
        # must happen before any cache checks, since the rewritten query may have a different cache key and embedding. 
        # This ensures that follow-up questions can benefit from caching even if the original question was not a cache hit.
        effective_query = query
        if history:
            on_thinking("Checking conversation history...")
            effective_query, rewrite_tokens = _rewrite_with_history(query, history, client)
            if effective_query != query:
                logger.info("[pipeline] Query rewritten: %r → %r", query, effective_query)
                on_thinking("Rewriting question with conversation history...")

        # Normalize and hash the (potentially rewritten) query for caching
        # Note if there isn't history the query is unchanged
        normalized_query = normalize_question(effective_query)
        question_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        verse_ref = (extract_verse_reference(effective_query) or [None])[0]

        # Option 1: Cache hit -- return cached answer without running pipeline (Fast and Cheapest)
        # Question hash matches exactly with a previous query that had no cache hit (i.e. was not previously served from cache)

        # -- Cache check -----------------------------------------------------
        on_thinking("Thinking...")

        exact_row = db.execute(
            select(QueryCache)
            .where(QueryCache.question_hash == question_hash, QueryCache.cache_hit == False)  # noqa: E712
            .limit(1)
        ).scalar_one_or_none()

        if exact_row:
            on_thinking("Found previous answer.")
            _insert_cache_row(
                db=db,
                question_raw=query,
                question_normalized=normalized_query,
                question_hash=question_hash,
                embedding=exact_row.embedding,
                response=exact_row.response,
                cache_hit=True,
                sources=exact_row.sources,
                cache_hit_type="exact",
                cache_source_id=exact_row.query_id,
                session_id=session_id,
                verse_reference=verse_ref,
            )
            _append_history(session_id, query, exact_row.response, db)
            return {"answer": exact_row.response, "sources": exact_row.sources or []}

        # Option 2: Checking for semantically similar cached questions asked using vector search (Fast and Cheaper than full pipeline)

        # -- Embed query ------------------------------------------------------
        query_embedding, embed_tokens = _embed_query(client, normalized_query, Config.EMBEDDING_MODEL)

        # -- Vector cache check -----------------------------------------------
        distance_expr = QueryCache.embedding.cosine_distance(query_embedding).label("distance")
        vcache_stmt = (
            select(QueryCache, distance_expr)
            .where(QueryCache.cache_hit == False)  # noqa: E712
            .order_by(distance_expr)
            .limit(1)
        )
        # If the question has a verse reference, we only want to compare against cached questions with the same verse reference. 
        # If it doesn't have a verse reference, we only want to compare against cached questions that also don't have a verse reference. This prevents us from accidentally returning a cached answer about a different verse that happens to have a similar embedding.
        if verse_ref is not None:
            vcache_stmt = vcache_stmt.where(QueryCache.verse_reference == verse_ref)
        else:
            # For questions without verse refs
            vcache_stmt = vcache_stmt.where(QueryCache.verse_reference.is_(None))

        vcache_row = db.execute(vcache_stmt).first()
        if vcache_row:
            cached, distance = vcache_row
            similarity = 1 - float(distance)
            if similarity > 0.85:
                on_thinking("Found similar answer in cache")
                _insert_cache_row(
                    db=db,
                    question_raw=query,
                    question_normalized=normalized_query,
                    question_hash=question_hash,
                    embedding=query_embedding,
                    response=cached.response,
                    cache_hit=True,
                    sources=cached.sources,
                    cache_hit_type="vector",
                    cache_source_id=cached.query_id,
                    similarity_score=similarity,
                        token_information={
                            "embedding": {
                                "hit": True,
                                "prompt_tokens": embed_tokens,
                            }
                        },
                    session_id=session_id,
                    verse_reference=verse_ref,
                )
                _append_history(session_id, query, cached.response, db)
                return {"answer": cached.response, "sources": cached.sources or []}

        # Option 3: No cache hit -- run full pipeline (Slowest and Most Expensive)

        # -- Tier 1: Document routing -----------------------------------------
        on_thinking("Selecting relevant sources...")
        selected_docs = _tier1_route_documents(
            effective_query, query_embedding, client, db
        )

        # If unable to find any relevant documents -- Early Exit
        if not selected_docs:
            answer = (
               "I'm sorry, I couldn't find any relevant sources in the library to answer your question. Please try rephrasing or asking about a different topic."
            )
            _insert_cache_row(
                db=db,
                question_raw=query,
                question_normalized=normalized_query,
                question_hash=question_hash,
                embedding=query_embedding,
                response=answer,
                cache_hit=False,
                sources=[],
                token_information={
                    "embedding": {
                        "hit": True,
                        "prompt_tokens": embed_tokens,
                    }
                },
                session_id=session_id,
                verse_reference=verse_ref,
            )
            return {"answer": answer, "sources": []}

        titles = ", ".join(d.title for d in selected_docs)
        on_thinking(f"Selected {len(selected_docs)} source(s): {titles}")

        # -- Tier 2 + extraction per document --------------------------------
        source_data = []  # list of {"doc", "section_title", "start", "end", "text"}

        for doc in selected_docs:
            on_thinking(f"Navigating table of contents for: {doc.title}")
            start_page, end_page, section_title = _tier2_navigate_section(effective_query, doc, client, db)

            on_thinking(f"Extracting pages {start_page}-{end_page} from: {doc.title}")
            raw_text = _extract_page_text(doc.document_id, start_page, end_page, db)

            source_data.append({
                "doc": doc,
                "section_title": section_title,
                "start_page": start_page,
                "end_page": end_page,
                "raw_text": raw_text,
            })

        # -- Synthesis --------------------------------------------------------
        on_thinking("Summarizing ...")

        distill_tokens = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        sources_out = [
            {
                "document_id": str(sd["doc"].document_id),
                "title": sd["doc"].title,
                "author": sd["doc"].author,
                "source": sd["doc"].source,
                "section_title": sd["section_title"],
                "pages": f"{sd['start_page']}-{sd['end_page']}",
            }
            for sd in source_data
        ]

        # TODO I feel like there is a function that could be clear for this
        # If there's only one source, we can feed the full text to the LLM. 
        # If there are multiple sources, we distill each one first to avoid hitting token limits, then feed the briefs to the LLM for final synthesis.
        if len(source_data) == 1:
            sd = source_data[0]
            page_count = sd["end_page"] - sd["start_page"] + 1

            if page_count > _DISTILL_THRESHOLD_PAGES or len(sd["raw_text"]) > _DISTILL_THRESHOLD_CHARS:
                on_thinking(f"Researching extract from: {sd['doc'].title}")
                context_text, distill_usage = _distill_source(
                    effective_query, sd["doc"].title, sd["section_title"], sd["raw_text"], client
                )
                distill_tokens["calls"] += 1
                distill_tokens["prompt_tokens"] += distill_usage["prompt_tokens"]
                distill_tokens["completion_tokens"] += distill_usage["completion_tokens"]
                distill_tokens["total_tokens"] += distill_usage["total_tokens"]
            else:
                context_text = (
                    f"[Source: {sd['doc'].title} -- {sd['section_title']}, "
                    f"pages {sd['start_page']}-{sd['end_page']}]\n\n{sd['raw_text']}"
                )

            user_message = f"Question: {effective_query}\n\nDocument excerpts:\n\n{context_text}"
        else:
            # Multi-source: distil each in parallel then synthesise
            briefs = {}

            def _distil(sd):
                page_count = sd["end_page"] - sd["start_page"] + 1
                if page_count > _DISTILL_THRESHOLD_PAGES or len(sd["raw_text"]) > _DISTILL_THRESHOLD_CHARS:
                    brief, usage = _distill_source(
                        effective_query, sd["doc"].title, sd["section_title"], sd["raw_text"], client
                    )
                else:
                    brief = (
                        f"[Source: {sd['doc'].title} -- {sd['section_title']}, "
                        f"pages {sd['start_page']}-{sd['end_page']}]\n\n{sd['raw_text']}"
                    )
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return sd, brief, usage

            # Distil in parallel to speed up processing of multiple large sources
            # TODO: Could we launch distillation tasks on a separate pod or something? I am used to doing this on OpenShift but not sure how to do this on Railway
            with ThreadPoolExecutor(max_workers=min(len(source_data), 4)) as pool:
                futures = {pool.submit(_distil, sd): sd for sd in source_data}
                for future in as_completed(futures):
                    sd, brief, usage = future.result()
                    on_thinking(f"Researching brief for: {sd['doc'].title}")
                    briefs[str(sd["doc"].document_id)] = brief
                    if usage["total_tokens"] > 0:
                        distill_tokens["calls"] += 1
                        distill_tokens["prompt_tokens"] += usage["prompt_tokens"]
                        distill_tokens["completion_tokens"] += usage["completion_tokens"]
                        distill_tokens["total_tokens"] += usage["total_tokens"]

            context_parts = [
                f"[Source: {sd['doc'].title} -- {sd['section_title']}, "
                f"pages {sd['start_page']}-{sd['end_page']}]\n\n{briefs[str(sd['doc'].document_id)]}"
                for sd in source_data
            ]
            context_text = "\n\n---\n\n".join(context_parts)
            user_message = f"Question: {effective_query}\n\nResearch briefs from multiple sources:\n\n{context_text}"

        completion = client.chat.completions.create(
            model=Config.COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *[{"role": m["role"], "content": m["content"]} for m in history],
                {"role": "user", "content": user_message},
            ],
            max_tokens=3500,
            temperature=0,
        )
        answer = completion.choices[0].message.content

        _insert_cache_row(
            db=db,
            question_raw=query,
            question_normalized=normalized_query,
            question_hash=question_hash,
            embedding=query_embedding,
            response=answer,
            cache_hit=False,
            sources=sources_out,
            token_information=(
                {
                    "embedding": {
                        "hit": True,
                        "prompt_tokens": embed_tokens,
                    },
                    "chat_completion": {
                        "hit": True,
                        "prompt_tokens": int(getattr(completion.usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(completion.usage, "completion_tokens", 0) or 0),
                        "total_tokens": int(getattr(completion.usage, "total_tokens", 0) or 0),
                    },
                    **(
                        {
                            "distill": {
                                "hit": True,
                                "calls": distill_tokens["calls"],
                                "prompt_tokens": distill_tokens["prompt_tokens"],
                                "completion_tokens": distill_tokens["completion_tokens"],
                                "total_tokens": distill_tokens["total_tokens"],
                            }
                        }
                        if distill_tokens["calls"] > 0
                        else {}
                    ),
                    **(
                        {
                            "rewrite": {
                                "hit": True,
                                **rewrite_tokens,
                            }
                        }
                        if rewrite_tokens is not None
                        else {}
                    ),
                }
            ),
            session_id=session_id,
            verse_reference=verse_ref,
        )

        _append_history(session_id, query, answer, db)
        return {"answer": answer, "sources": sources_out}
    finally:
        client.close()


async def answer_question(
    query: str,
    api_key: str | None,
    db: Session,
    session_id: str,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields SSE-ready dicts.

    Each dict has keys {"event": str, "data": dict} matching the wire format
    consumed by sse-starlette's EventSourceResponse.

    Yields:
        {"event": "thinking", "data": {"message": "..."}}   -- pipeline stage updates
        {"event": "answer",   "data": {"text": "...", "sources": [...]}}
        {"event": "error",    "data": {"message": "..."}}   -- on failure
    """
    thinking_messages: list[str] = []

    # TODO: Should we put this as a reuseable module
    def _on_thinking(msg: str) -> None:
        thinking_messages.append(msg)
        # We can't yield from a sync callback; we'll flush in the async wrapper below

    # We run the synchronous pipeline in a thread so it doesn't block the event loop,
    # but we need to surface thinking messages as they happen. We do this by running
    # the pipeline step by step -- since _on_thinking appends to a list we can poll.
    import asyncio
    loop = asyncio.get_event_loop()

    result_container: list[dict] = []
    error_container: list[Exception] = []
    emitted_index = 0

    def _pipeline_thread():
        try:
            result = _run_pipeline(
                query=query,
                api_key=api_key,
                db=db,
                session_id=session_id,
                on_thinking=_on_thinking,
            )
            result_container.append(result)
        except Exception as exc:
            error_container.append(exc)

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, _pipeline_thread)

    # Poll until the pipeline thread completes, flushing new thinking messages each tick
    while not future.done():
        await asyncio.sleep(0.1)
        while emitted_index < len(thinking_messages):
            yield {"event": "thinking", "data": json.dumps({"message": thinking_messages[emitted_index]})}
            emitted_index += 1

    # Flush any remaining messages
    while emitted_index < len(thinking_messages):
        yield {"event": "thinking", "data": json.dumps({"message": thinking_messages[emitted_index]})}
        emitted_index += 1

    if error_container:
        exc = error_container[0]
        logger.error("Pipeline error during SSE stream: %s", exc)
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return

    result = result_container[0]
    yield {"event": "answer", "data": json.dumps({"text": result["answer"], "sources": result["sources"]})}


