import io
import logging

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from file bytes without writing to disk."""
    name = (filename or "").lower()

    if name.endswith(".pdf"):
        logger.info("Extracting text from PDF: %s", filename)
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif name.endswith(".docx"):
        logger.info("Extracting text from DOCX: %s", filename)
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif name.endswith(".txt"):
        logger.info("Extracting text from TXT: %s", filename)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1")

    else:
        raise ValueError(f"Unsupported file type: {name}")
