import hashlib
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import AsyncGenerator

import openai
import spacy
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, DocumentStructure, DocumentPage, QueryCache
from app.services.openai_client import create_openai_client
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

SYSTEM_PROMPT = """You are a Christian theological assistant helping users understand the Bible through the provided sources.

Answer questions using ONLY the document excerpts provided. Do not draw on outside knowledge.
- Cite every claim using the format: (Title, Section Title, pages X-Y)
- Present differing interpretations fairly if they appear in the source material

When the question refers to a specific verse or verse range:
- If the provided excerpts directly address that verse or range, answer from those excerpts
- If the excerpts do not contain enough detail about that specific verse but do cover the surrounding chapter or passage, provide an answer based on that broader context and begin your response with a brief note such as: "I couldn't find specific commentary on that verse, but here is what the sources say about the surrounding passage:"
- If the excerpts contain no relevant information at all, say so clearly -- do not speculate or fill in gaps

Format your response using Markdown: use **bold** for emphasis, headings (##, ###) to organise longer answers, and bullet lists where appropriate.
Write with clarity and pastoral warmth, grounded entirely in the provided documents."""

DISTILL_PROMPT = """You are a research assistant summarising theological commentary for use in a comparative synthesis.

Given the document excerpt below, write a focused research brief (3-6 paragraphs) that:
- Captures the author's main argument and key insights relevant to the query
- Preserves important quotations or specific exegetical points
- Notes the author's theological framework where relevant
- Is written in third-person ("The author argues...", "Henry notes...")

Do not add information not found in the excerpt. Be concise but thorough."""

# How many pages constitute a "large" section that warrants pre-distillation
_DISTILL_THRESHOLD_PAGES = 15
# Fallback for single-page URL ingests: distil if raw text exceeds this (~ 3,000 tokens)
_DISTILL_THRESHOLD_CHARS = 12_000


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


# ---------------------------------------------------------------------------
# Verse reference extraction
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tier 1 -- Document Router
# ---------------------------------------------------------------------------

def _embed_query(client: openai.OpenAI, text: str) -> tuple[list[float], int]:
    resp = client.embeddings.create(model=Config.EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding, resp.usage.prompt_tokens


def _tier1_route_documents(
    query: str,
    query_embedding: list[float],
    document_ids_override: list[str] | None,
    client: openai.OpenAI,
    db: Session,
) -> list[Document]:
    """Return the list of Document objects to search for this query."""
    if document_ids_override:
        docs = []
        for did in document_ids_override:
            doc = db.get(Document, uuid.UUID(did))
            if doc:
                docs.append(doc)
        return docs

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

    routing_prompt = (
        "You are a theological research librarian. A user has asked the following question:\n\n"
        f"QUESTION: {query}\n\n"
        "Below are the available sources. Select the document IDs most likely to contain a "
        "relevant answer. Return ONLY a JSON array of document_id strings -- no explanation.\n\n"
        f"{doc_summaries}\n\n"
        "Return format: [\"uuid1\", \"uuid2\"]"
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

    nav_prompt = (
        f"You are navigating the table of contents of '{doc.title}' to answer this question:\n\n"
        f"QUESTION: {query}\n\n"
        f"TOC:\n{toc_text}\n\n"
        "Select the single MOST relevant TOC entry. Prefer the most specific (deepest level) "
        "entry that covers the topic. Return ONLY a JSON object with keys 'index' (1-based) "
        "and 'section_title'. No explanation.\n"
        'Return format: {"index": 3, "section_title": "..."}'
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
# Page extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Distillation (per-source summary for large extracts or multi-source)
# ---------------------------------------------------------------------------

# Hard cap on input to a single distillation call (~20,000 tokens at 4 chars/token).
# Prevents 429s when a section is very large. Add a TOC to avoid hitting this limit.
_DISTILL_MAX_INPUT_CHARS = 80_000


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


# ---------------------------------------------------------------------------
# Core pipeline (shared by answer_question and answer_question_stream)
# ---------------------------------------------------------------------------

def _run_pipeline(
    query: str,
    document_ids: list[str] | None,
    api_key: str | None,
    db: Session,
    session_id: str,
    on_thinking: callable,  # fn(message: str) -> None
) -> dict:
    """Execute the full Tier1->Tier2->extract->synthesise pipeline.

    *on_thinking* is called synchronously at each stage with a human-readable
    status string, enabling SSE streaming without duplicating pipeline logic.

    Returns {"answer": str, "sources": list[dict]}.
    """
    client = create_openai_client(api_key)
    try:
        normalized_query = normalize_question(query)
        question_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
        verse_ref = (extract_verse_reference(query) or [None])[0]

        # Option 1: Cache hit -- return cached answer without running pipeline (Fast and Cheapest)
        # Question hash matches exactly with a previous query that had no cache hit (i.e. was not previously served from cache)

        # -- Cache check -----------------------------------------------------
        on_thinking("Checking query cache...")

        exact_row = db.execute(
            select(QueryCache)
            .where(QueryCache.question_hash == question_hash, QueryCache.cache_hit == False)  # noqa: E712
            .limit(1)
        ).scalar_one_or_none()

        if exact_row:
            on_thinking("Found cached answer.")
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
            return {"answer": exact_row.response, "sources": exact_row.sources or []}

        # Option 2: Checking for semantically similar cached questions asked using vector search (Fast and Cheaper than full pipeline)

        # -- Embed query ------------------------------------------------------
        query_embedding, embed_tokens = _embed_query(client, normalized_query)

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
                on_thinking("Found semantically similar cached answer.")
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
                return {"answer": cached.response, "sources": cached.sources or []}

        # Option 3: No cache hit -- run full pipeline (Slowest and Most Expensive)

        # -- Tier 1: Document routing -----------------------------------------
        on_thinking("Selecting relevant sources...")
        selected_docs = _tier1_route_documents(
            query, query_embedding, document_ids, client, db
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
            start_page, end_page, section_title = _tier2_navigate_section(query, doc, client, db)

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
        on_thinking("Synthesizing final answer...")

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

        # If there's only one source, we can feed the full text to the LLM. 
        # If there are multiple sources, we distill each one first to avoid hitting token limits, then feed the briefs to the LLM for final synthesis.
        if len(source_data) == 1:
            sd = source_data[0]
            page_count = sd["end_page"] - sd["start_page"] + 1

            if page_count > _DISTILL_THRESHOLD_PAGES or len(sd["raw_text"]) > _DISTILL_THRESHOLD_CHARS:
                on_thinking(f"Distilling large extract from: {sd['doc'].title}")
                context_text, distill_usage = _distill_source(
                    query, sd["doc"].title, sd["section_title"], sd["raw_text"], client
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

            user_message = f"Question: {query}\n\nDocument excerpts:\n\n{context_text}"
        else:
            # Multi-source: distil each in parallel then synthesise
            briefs = {}

            def _distil(sd):
                page_count = sd["end_page"] - sd["start_page"] + 1
                if page_count > _DISTILL_THRESHOLD_PAGES or len(sd["raw_text"]) > _DISTILL_THRESHOLD_CHARS:
                    brief, usage = _distill_source(
                        query, sd["doc"].title, sd["section_title"], sd["raw_text"], client
                    )
                else:
                    brief = (
                        f"[Source: {sd['doc'].title} -- {sd['section_title']}, "
                        f"pages {sd['start_page']}-{sd['end_page']}]\n\n{sd['raw_text']}"
                    )
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                return sd, brief, usage

            # Distil in parallel to speed up processing of multiple large sources
            # TODO: Could we lanch distillation tasks on a seprate pod or something? I am used to doing this on openshift but not sure how to do this on railway
            with ThreadPoolExecutor(max_workers=min(len(source_data), 4)) as pool:
                futures = {pool.submit(_distil, sd): sd for sd in source_data}
                for future in as_completed(futures):
                    sd, brief, usage = future.result()
                    on_thinking(f"Distilled research brief for: {sd['doc'].title}")
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
            user_message = f"Question: {query}\n\nResearch briefs from multiple sources:\n\n{context_text}"

        completion = client.chat.completions.create(
            model=Config.COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
                }
            ),
            session_id=session_id,
            verse_reference=verse_ref,
        )

        return {"answer": answer, "sources": sources_out}
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Public API -- synchronous
# ---------------------------------------------------------------------------

def answer_question(
    query: str,
    document_ids: list[str] | None = None,
    api_key: str | None = None,
    db: Session | None = None,
    session_id: str | None = None,
) -> dict:
    return _run_pipeline(
        query=query,
        document_ids=document_ids,
        api_key=api_key,
        db=db,
        session_id=session_id or "",
        on_thinking=lambda msg: logger.info("[pipeline] %s", msg),
    )


# ---------------------------------------------------------------------------
# Public API -- SSE async generator
# ---------------------------------------------------------------------------

async def answer_question_stream(
    query: str,
    document_ids: list[str] | None,
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
                document_ids=document_ids,
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


