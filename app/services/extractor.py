import io
import logging

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Extract text per page from file bytes. Returns list of (page_number, text) tuples (1-indexed).

    For DOCX and TXT files, the entire content is returned as a single page.
    """
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        logger.info("Extracting pages from PDF: %s", filename)
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((i, text))
            return pages

    elif name.endswith(".docx"):
        logger.info("Extracting text from DOCX: %s", filename)
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(1, text)]

    elif name.endswith(".txt"):
        logger.info("Extracting text from TXT: %s", filename)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        return [(1, text)]

    else:
        raise ValueError(f"Unsupported file type: {name}")
