import os

import httpx
import openai


DEFAULT_OPENAI_TIMEOUT_SECONDS = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))


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