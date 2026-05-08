import logging
import os

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.services.openai_client import create_openai_client

logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "10"))
EMBEDDING_MODEL = "text-embedding-3-small"


@retry(
    retry=retry_if_exception_type(
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        )
    ),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch(client: openai.OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data], response.usage.total_tokens
    except Exception as exc:
        logger.error(
            "Embedding call failed: %s: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        raise


def embed_chunks(chunks: list[dict], api_key: str) -> tuple[list[list[float]], int]:
    client = create_openai_client(api_key)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = []
    total_tokens = 0

    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            logger.info(
                "Embedding batch %d–%d of %d chunks", i + 1, i + len(batch), len(texts)
            )
            batch_embeddings, batch_tokens = _embed_batch(client, batch)
            embeddings.extend(batch_embeddings)
            total_tokens += batch_tokens
    finally:
        client.close()

    return embeddings, total_tokens
