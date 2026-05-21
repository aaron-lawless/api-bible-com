import logging
import os

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from app.config import Config
from app.services.llm.openai_client import create_openai_client

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = Config.OPENAI_EMBEDDING_MODEL


# TODO look at the embedding within the client file to see if we can unify the functions

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
def embed_text(text: str, api_key: str) -> list[float]:
    """Embed a single piece of text and return the embedding vector."""
    client = create_openai_client(api_key)
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        return response.data[0].embedding
    finally:
        client.close()
