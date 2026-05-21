import os

import httpx
import openai


DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))

# TODO: Should the creation of the client be created once and reused?
def create_openai_client(
    api_key: str | None,
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
) -> openai.OpenAI:
    # max_retries=0: disable openai's internal retry so tenacity in embedder.py
    # is the single source of retry logic (avoids stacking retry delays).
    try:
        return openai.OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    except TypeError as exc:
        if "unexpected keyword argument 'proxies'" not in str(exc):
            raise

        return openai.OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
            http_client=httpx.Client(timeout=timeout_seconds),
        )
    
# TODO look at the embedder file and look for the similar function and consolidate them.
def _embed_query(client: openai.OpenAI, text: str, model: str) -> tuple[list[float], int]:
    resp = client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding, resp.usage.prompt_tokens