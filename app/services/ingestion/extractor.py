import io
import logging
import re

import fitz  # PyMuPDF
import pymupdf4llm
from docx import Document

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

            text = text.strip()
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
