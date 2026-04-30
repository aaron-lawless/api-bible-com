import logging

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_text(text: str) -> list[dict]:
    enc = tiktoken.get_encoding("cl100k_base")

    def token_length(s: str) -> int:
        return len(enc.encode(s))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=token_length,
    )

    raw_chunks = splitter.split_text(text)
    logger.info("Split text into %d chunks", len(raw_chunks))

    return [{"content": chunk, "chunk_index": i} for i, chunk in enumerate(raw_chunks)]
