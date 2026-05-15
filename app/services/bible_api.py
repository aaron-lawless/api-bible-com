"""
Client for the Free Use Bible API (https://bible.helloao.org).

Used to fetch verse text for query expansion — appending the actual verse
content to the search query embedding improves retrieval quality when
document chunks quote or paraphrase the verse text.

Chapter responses are cached in-memory for the lifetime of the process;
verse text never changes so there is no need for cache invalidation.
"""

import logging
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://bible.helloao.org/api"
_DEFAULT_TRANSLATION = "BSB"  # Berean Standard Bible — free, reliable, widely used

# Map from lowercase book names (including common abbreviations / variants) to
# USFM standard book IDs used by the API.
_BOOK_NAME_TO_ID: dict[str, str] = {
    # Old Testament
    "genesis": "GEN", "gen": "GEN",
    "exodus": "EXO", "exo": "EXO", "exod": "EXO",
    "leviticus": "LEV", "lev": "LEV",
    "numbers": "NUM", "num": "NUM",
    "deuteronomy": "DEU", "deu": "DEU", "deut": "DEU",
    "joshua": "JOS", "jos": "JOS", "josh": "JOS",
    "judges": "JDG", "jdg": "JDG", "judg": "JDG",
    "ruth": "RUT", "rut": "RUT",
    "1 samuel": "1SA", "1samuel": "1SA", "1sa": "1SA", "1 sam": "1SA",
    "2 samuel": "2SA", "2samuel": "2SA", "2sa": "2SA", "2 sam": "2SA",
    "1 kings": "1KI", "1kings": "1KI", "1ki": "1KI",
    "2 kings": "2KI", "2kings": "2KI", "2ki": "2KI",
    "1 chronicles": "1CH", "1chronicles": "1CH", "1ch": "1CH", "1 chron": "1CH",
    "2 chronicles": "2CH", "2chronicles": "2CH", "2ch": "2CH", "2 chron": "2CH",
    "ezra": "EZR", "ezr": "EZR",
    "nehemiah": "NEH", "neh": "NEH",
    "esther": "EST", "est": "EST",
    "job": "JOB",
    "psalms": "PSA", "psalm": "PSA", "psa": "PSA", "ps": "PSA",
    "proverbs": "PRO", "pro": "PRO", "prov": "PRO",
    "ecclesiastes": "ECC", "ecc": "ECC", "eccl": "ECC",
    "song of solomon": "SNG", "song": "SNG", "sng": "SNG", "sos": "SNG",
    "isaiah": "ISA", "isa": "ISA",
    "jeremiah": "JER", "jer": "JER",
    "lamentations": "LAM", "lam": "LAM",
    "ezekiel": "EZK", "ezk": "EZK", "ezek": "EZK",
    "daniel": "DAN", "dan": "DAN",
    "hosea": "HOS", "hos": "HOS",
    "joel": "JOL", "jol": "JOL",
    "amos": "AMO", "amo": "AMO",
    "obadiah": "OBA", "oba": "OBA",
    "jonah": "JON", "jon": "JON",
    "micah": "MIC", "mic": "MIC",
    "nahum": "NAM", "nam": "NAM",
    "habakkuk": "HAB", "hab": "HAB",
    "zephaniah": "ZEP", "zep": "ZEP", "zeph": "ZEP",
    "haggai": "HAG", "hag": "HAG",
    "zechariah": "ZEC", "zec": "ZEC", "zech": "ZEC",
    "malachi": "MAL", "mal": "MAL",
    # New Testament
    "matthew": "MAT", "mat": "MAT", "matt": "MAT",
    "mark": "MRK", "mrk": "MRK",
    "luke": "LUK", "luk": "LUK",
    "john": "JHN", "jhn": "JHN",
    "acts": "ACT", "act": "ACT",
    "romans": "ROM", "rom": "ROM",
    "1 corinthians": "1CO", "1corinthians": "1CO", "1co": "1CO", "1 cor": "1CO",
    "2 corinthians": "2CO", "2corinthians": "2CO", "2co": "2CO", "2 cor": "2CO",
    "galatians": "GAL", "gal": "GAL",
    "ephesians": "EPH", "eph": "EPH",
    "philippians": "PHP", "php": "PHP", "phil": "PHP",
    "colossians": "COL", "col": "COL",
    "1 thessalonians": "1TH", "1thessalonians": "1TH", "1th": "1TH", "1 thess": "1TH",
    "2 thessalonians": "2TH", "2thessalonians": "2TH", "2th": "2TH", "2 thess": "2TH",
    "1 timothy": "1TI", "1timothy": "1TI", "1ti": "1TI", "1 tim": "1TI",
    "2 timothy": "2TI", "2timothy": "2TI", "2ti": "2TI", "2 tim": "2TI",
    "titus": "TIT", "tit": "TIT",
    "philemon": "PHM", "phm": "PHM",
    "hebrews": "HEB", "heb": "HEB",
    "james": "JAS", "jas": "JAS",
    "1 peter": "1PE", "1peter": "1PE", "1pe": "1PE", "1 pet": "1PE",
    "2 peter": "2PE", "2peter": "2PE", "2pe": "2PE", "2 pet": "2PE",
    "1 john": "1JN", "1john": "1JN", "1jn": "1JN",
    "2 john": "2JN", "2john": "2JN", "2jn": "2JN",
    "3 john": "3JN", "3john": "3JN", "3jn": "3JN",
    "jude": "JUD", "jud": "JUD",
    "revelation": "REV", "rev": "REV",
}


def _extract_text_from_content(content_items: list) -> str:
    """Flatten the API's content array into a plain text string."""
    parts = []
    for item in content_items:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            # FormattedText has a 'text' key
            if "text" in item:
                parts.append(item["text"])
            # InlineHeading has a 'heading' key — skip it for verse text
    return " ".join(parts)


@lru_cache(maxsize=512)
def _fetch_chapter(book_id: str, chapter: int, translation: str) -> dict | None:
    """Fetch and cache a chapter from the Bible API. Returns the raw chapter dict or None on error."""
    url = f"{_BASE_URL}/{translation}/{book_id}/{chapter}.json"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.warning("Bible API fetch failed for %s: %s", url, exc)
        return None


def get_verse_text(
    book_name: str,
    chapter: int,
    verse_start: int,
    verse_end: int | None = None,
    translation: str = _DEFAULT_TRANSLATION,
) -> str | None:
    """Return the plain text of a verse or verse range, or None if unavailable.

    Args:
        book_name: Lowercase book name as extracted from the query (e.g. "romans").
        chapter: Chapter number.
        verse_start: First verse number.
        verse_end: Last verse number (inclusive). If None, fetches a single verse.
        translation: Bible translation ID (default BSB).
    """
    book_id = _BOOK_NAME_TO_ID.get(book_name.lower())
    if not book_id:
        logger.debug("No book ID mapping found for: %r", book_name)
        return None

    chapter_data = _fetch_chapter(book_id, chapter, translation)
    if not chapter_data:
        return None

    end = verse_end if verse_end is not None else verse_start
    verse_texts = []

    for item in chapter_data.get("chapter", {}).get("content", []):
        if isinstance(item, dict) and item.get("type") == "verse":
            verse_num = item.get("number")
            if verse_num is not None and verse_start <= verse_num <= end:
                text = _extract_text_from_content(item.get("content", []))
                if text:
                    verse_texts.append(text)

    return " ".join(verse_texts) if verse_texts else None
