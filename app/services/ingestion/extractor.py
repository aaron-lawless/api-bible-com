import io
import json
import logging
import re

import fitz  # PyMuPDF
import openai
import pymupdf4llm
from docx import Document
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# Map DOCX paragraph style names to markdown heading markers
_HEADING_STYLE_MAP = {
    "Title": "#",
    "Subtitle": "##",
    "Heading 1": "#",
    "Heading 2": "##",
    "Heading 3": "###",
    "Heading 4": "####",
}

# Regex to find markdown headings: # … through ####
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# pymupdf4llm emits this placeholder for every image it skips
_IMAGE_PLACEHOLDER_RE = re.compile(r"\*\*==>\s+picture\s+\[\d+\s*x\s*\d+\]\s+intentionally omitted", re.IGNORECASE)


def _docx_to_markdown(data: bytes) -> str:
    """Convert DOCX bytes to markdown using paragraph styles for heading detection."""
    doc = Document(io.BytesIO(data))
    lines = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        for style_prefix, marker in _HEADING_STYLE_MAP.items():
            if para.style.name.startswith(style_prefix):
                lines.append(f"{marker} {text}")
                break
        else:
            if "List" in para.style.name:
                lines.append(f"- {text}")
            else:
                lines.append(text)
    return "\n\n".join(lines)


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Extract text per page from file bytes as markdown.

    Returns list of (page_number, text) tuples (1-indexed).
    PDF pages are extracted individually; DOCX and TXT are returned as a single page.
    """
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        logger.info("Extracting pages from PDF: %s", filename)
        with fitz.open(stream=data, filetype="pdf") as doc:
            chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

        pages: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks, start=1):
            page_number = idx
            text = ""

            if isinstance(chunk, dict):
                text = str(chunk.get("text") or chunk.get("markdown") or "")
                metadata = chunk.get("metadata") or {}
                raw_page = metadata.get("page")
                if raw_page is None:
                    raw_page = metadata.get("page_number")
                if isinstance(raw_page, int):
                    # `page` is typically 0-based; `page_number` is often 1-based.
                    page_number = raw_page + 1 if raw_page < 1 else raw_page
            elif isinstance(chunk, str):
                text = chunk

            text = _IMAGE_PLACEHOLDER_RE.sub("", text).strip()
            if text:
                pages.append((page_number, text))

        if not pages:
            raise ValueError("No text could be extracted from the PDF")

        return pages

    elif name.endswith(".docx"):
        logger.info("Extracting text from DOCX: %s", filename)
        return [(1, _docx_to_markdown(data))]

    elif name.endswith(".txt"):
        logger.info("Extracting text from TXT: %s", filename)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return [(1, text)]

    else:
        raise ValueError(f"Unsupported file type: {name}")


def build_toc_from_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Parse markdown headings from extracted pages and return TOC entries.

    Returns a list of dicts with keys: section_title, start_page, end_page, level.
    level 1 = H1 (#), level 2 = H2 (##), etc.
    Each section's end_page extends until the next heading at the same or higher level.
    Returns an empty list if no headings are found.
    """
    headings: list[tuple[int, int, str]] = []
    for page_num, text in pages:
        for match in _HEADING_RE.finditer(text):
            headings.append((page_num, len(match.group(1)), match.group(2).strip()))

    if not headings:
        return []

    total_pages = max(p for p, _ in pages) if pages else 1

    entries = []
    for i, (page_num, level, title) in enumerate(headings):
        end_page = total_pages
        for next_page, next_level, _ in headings[i + 1:]:
            if next_level <= level:
                end_page = max(next_page - 1, page_num)
                break
        entries.append({
            "section_title": title,
            "start_page": page_num,
            "end_page": end_page,
            "level": level,
        })

    return entries

# This is an optional flag to enable the AI-generated TOC endpoint
def build_toc_from_ai(
    pages: list[tuple[int, str]],
    title: str,
    api_key: str | None,
    model: str = "gpt-4o",
    batch_size: int = 10,
) -> list[dict]:
    """Use OpenAI to generate a Table of Contents from page text.

    Pages are processed in batches of `batch_size` to stay within context limits.
    If a batch exceeds the model's context window it is automatically split in half
    and retried until the batch is a single page, at which point the error is raised.
    TOC entries from all batches are merged and returned as a single sorted list.

    Returns a list of TOC entry dicts (same format as build_toc_from_pages).
    Raises on unrecoverable API errors or if a single-page batch still exceeds context.
    """
    from app.services.llm.openai_client import create_openai_client

    if not pages:
        return []

    total_pages = max(p for p, _ in pages)
    client = create_openai_client(api_key)
    # TODO: Move this to prompts.py
    system_prompt = (
        "You are a document analyst. Given a batch of pages from a document, identify every section, "
        "chapter, or heading that starts within this batch and return a Table of Contents. "
        "Respond with ONLY a valid JSON object containing a single key \"toc\" whose value is an array. "
        "Each element must have exactly these keys: "
        '"section_title" (string), "start_page" (integer), "end_page" (integer), "level" (integer 1-4). '
        "level 1 = top-level chapter, 2 = section, 3 = subsection, 4 = sub-subsection. "
        f"The overall document has {total_pages} pages total. "
        "Ensure start_page <= end_page and all page numbers are within the document range."
    )
    #TODO: This code is messy, could be refactored to be cleaner and more modular, but it works for now and we can improve it later if needed.
    # We want to wait and retry based on the wait time that OpenAI providers in the error message
    # Could we use their batch service instead?
    # Could we also not have it hit every page but instead have it create the markdown and fill in the gaps?
    def _call_batch(batch: list[tuple[int, str]]) -> list[dict]:
        first_page = batch[0][0]
        last_page = batch[-1][0]
        pages_text = "\n\n".join(f"Page {p}:\n{t}" for p, t in batch)
        user_prompt = (
            f'Document title: "{title}"\n'
            f"Batch: pages {first_page}–{last_page} of {total_pages}\n\n"
            f"{pages_text}"
        )

        from tenacity import RetryCallState

        _retry_wait_re = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)

        def _wait_strategy(retry_state: RetryCallState) -> float:
            exc = retry_state.outcome.exception()
            if isinstance(exc, openai.RateLimitError):
                match = _retry_wait_re.search(str(exc))
                if match:
                    suggested = float(match.group(1))
                    return suggested + 1.0
            # fallback: exponential backoff for all other retryable errors
            return min(4 * (2 ** (retry_state.attempt_number - 1)), 60)

        def _before_retry(retry_state: RetryCallState) -> None:
            exc = retry_state.outcome.exception()
            wait_secs = _wait_strategy(retry_state)
            logger.warning(
                "AI TOC batch pages %d\u2013%d: retrying (attempt %d) in %.1fs after %s: %s",
                first_page, last_page, retry_state.attempt_number, wait_secs, type(exc).__name__, exc,
            )

        @retry(
            retry=retry_if_exception_type(
                (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                    json.JSONDecodeError,
                )
            ),
            wait=_wait_strategy,
            stop=stop_after_attempt(5),
            before_sleep=_before_retry,
            reraise=True,
        )
        def _call_with_retry() -> list[dict]:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)

            if isinstance(parsed, list):
                raw_entries = parsed
            elif isinstance(parsed, dict):
                raw_entries = next((v for v in parsed.values() if isinstance(v, list)), [])
            else:
                return []

            valid: list[dict] = []
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    row = {
                        "section_title": str(entry["section_title"]).strip(),
                        "start_page": int(entry["start_page"]),
                        "end_page": int(entry["end_page"]),
                        "level": max(1, min(4, int(entry.get("level", 1)))),
                    }
                except (KeyError, ValueError, TypeError):
                    continue
                if not row["section_title"]:
                    continue
                if row["start_page"] < 1 or row["end_page"] < row["start_page"]:
                    continue
                row["end_page"] = min(row["end_page"], total_pages)
                valid.append(row)
            return valid

        return _call_with_retry()

    def _process_batch(batch: list[tuple[int, str]]) -> list[dict]:
        """Call the model for a batch, halving it automatically on context-window errors."""
        try:
            return _call_batch(batch)
        except openai.BadRequestError as exc:
            err_lower = str(exc).lower()
            if "context" not in err_lower and "token" not in err_lower and "length" not in err_lower:
                raise
            if len(batch) == 1:
                logger.error(
                    "AI TOC: single page %d still exceeds context window for '%s': %s",
                    batch[0][0], title, exc,
                )
                raise
            half = len(batch) // 2
            logger.warning(
                "AI TOC: context window exceeded for pages %d\u2013%d, splitting into halves of %d and %d",
                batch[0][0], batch[-1][0], half, len(batch) - half,
            )
            return _process_batch(batch[:half]) + _process_batch(batch[half:])

    all_entries: list[dict] = []
    batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
    logger.info(
        "AI TOC: processing '%s' in %d batch(es) of up to %d pages",
        title, len(batches), batch_size,
    )

    for batch_num, batch in enumerate(batches, start=1):
        entries = _process_batch(batch)
        logger.info(
            "AI TOC batch %d/%d: %d entries (pages %d–%d)",
            batch_num, len(batches), len(entries), batch[0][0], batch[-1][0],
        )
        all_entries.extend(entries)

    all_entries.sort(key=lambda e: e["start_page"])
    logger.info("AI TOC total: %d entries for '%s'", len(all_entries), title)
    return all_entries
