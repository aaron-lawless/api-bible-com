import logging

import httpx
import trafilatura

logger = logging.getLogger(__name__)


def scrape_url(url: str, timeout: int = 30) -> str:
    """Fetch *url* and return clean article text.

    Uses trafilatura for main-content extraction after a plain httpx GET.
    Raises ValueError if the fetch fails or no meaningful content is found.
    """
    logger.info("Scraping URL: %s", url)
    try:
        response = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AiBibleBot/1.0)"},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ValueError(
            f"HTTP {exc.response.status_code} fetching URL: {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise ValueError(f"Failed to fetch URL '{url}': {exc}") from exc

    html = response.text
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        output_format="markdown",
    )

    if not text or not text.strip():
        raise ValueError(f"No meaningful content could be extracted from: {url}")

    logger.info("Scraped %d characters from %s", len(text), url)
    return text.strip()
