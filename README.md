# AI Bible - RAG API

A document ingestion and semantic search API for Christian theological content. Accepts PDF uploads or web article URLs, stores page-by-page text with a table of contents, and answers natural language questions using a two-tier hierarchical routing pipeline powered by GPT-4o.

## Tech Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy 2.0
- **Database**: PostgreSQL with pgvector
- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Completions**: `gpt-4o`
- **Text extraction**: pdfplumber (PDF), python-docx (DOCX), trafilatura (web URLs)
- **Streaming**: sse-starlette (Server-Sent Events)
- **Deployment**: Railway

---

## RAG Pipeline

Questions are answered through a two-tier hierarchical routing pipeline. Each tier uses GPT-4o to make a routing decision before any text is read.

### Tier 1 - Document Router

The query is embedded and compared against all document `summary_embedding` vectors via cosine similarity. The top candidates are passed to GPT-4o, which selects the document IDs most likely to contain a relevant answer. If specific `document_ids` are provided in the request, Tier 1 is skipped.

### Tier 2 - Section Navigator

For each selected document, GPT-4o is shown the table of contents (section titles and page ranges) and picks the single most relevant section. If no TOC has been added for a document, the full page range is used.

### Page Extraction + Synthesis

The raw text for the chosen page range is fetched from `document_pages`. For large extracts (>15 pages) or multiple sources, GPT-4o first writes a focused research brief per source, then synthesises a final answer across all briefs.

### SSE Streaming

`GET /search/stream` streams each pipeline stage as a Server-Sent Event so the UI can display live progress ("Selecting relevant sources...", "Navigating table of contents for: ...", etc.) before the final answer arrives.

---

## Query Caching

Every question passes through two cache layers before running the full pipeline.

### Step 1 - Exact Match (zero API calls)

The question is normalised (lowercased, stop words removed, lemmatized via spaCy) and hashed with SHA-256. If the same hash exists from a previous non-cached answer, the stored response is returned immediately.

### Step 2 - Vector Match (1 embedding call, no completion call)

If no exact match is found, the question is embedded and compared via cosine similarity against previous answers. If the closest match scores above **0.85**, the existing response is reused.

### Step 3 - Full Pipeline

If no cache hit is found, the full Tier 1 -> Tier 2 -> extract -> synthesise pipeline runs and the result is stored for future reuse.

### Cache Record

Each query is recorded in `query_cache` with:

| Field | Description |
|---|---|
| `question_raw` | Original question text |
| `question_normalized` | Normalised form used for hashing |
| `question_hash` | SHA-256 of the normalised question |
| `embedding` | Vector embedding of the normalised question |
| `response` | The answer returned to the user |
| `sources` | List of `{document_id, title, author, source, section_title, pages}` |
| `cache_hit` | Whether this was served from cache |
| `cache_hit_type` | `"exact"`, `"vector"`, or `null` |
| `cache_source_id` | ID of the originating row (for cache hits) |
| `similarity_score` | Cosine similarity (vector hits only) |
| `token_information` | Embedding and/or completion token counts |
| `session_id` | Client session identifier |
| `verse_reference` | Canonical verse e.g. `"romans 8:1-2"`, null for general questions |

**Example `sources` value:**
```json
[
  {
    "document_id": "a3f1c2d4-8b5e-4f2a-9c1d-7e6f3a2b1c0d",
    "title": "Systematic Theology",
    "author": "Wayne Grudem",
    "source": null,
    "section_title": "Chapter 24: Justification",
    "pages": "512-534"
  }
]
```

---

## API Endpoints

### Search

| Method | Path | Description |
|---|---|---|
| `POST` | `/search` | Ask a question. Returns an answer and sources. Hits cache or runs the full pipeline. |
| `GET` | `/search/stream` | Same as POST /search but streams pipeline progress via Server-Sent Events. |
| `GET` | `/questions` | Paginated list of previously answered questions. |
| `GET` | `/questions/{query_id}` | Retrieve the full answer for a specific question by ID. No API call made. |

**`POST /search` body:**

| Field | Required | Description |
|---|---|---|
| `query` | Yes | The question to answer |
| `document_ids` | No | Array of UUIDs to restrict search to specific documents |

**`GET /search/stream` query parameters:**

| Parameter | Required | Description |
|---|---|---|
| `query` | Yes | The question to answer |
| `document_ids` | No | One or more UUIDs to restrict search to specific documents |

**`GET /questions` query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `page` | `1` | Page number |
| `page_size` | `20` | Results per page (max 100) |
| `search` | - | Optional case-insensitive text filter on the question |

---

### Documents

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents` | List all ingested documents (newest first). |
| `GET` | `/documents/{doc_id}` | Get a document and its structure count. |
| `DELETE` | `/documents/{doc_id}` | Delete a document and all its pages and TOC entries. |
| `GET` | `/documents/{doc_id}/structures` | List all TOC entries for a document, ordered by start page. |
| `POST` | `/documents/{doc_id}/structures` | Add a TOC entry to a document. |
| `PUT` | `/documents/{doc_id}/structures/{struct_id}` | Update a TOC entry. |
| `DELETE` | `/documents/{doc_id}/structures/{struct_id}` | Delete a TOC entry. |

**`POST /documents/{doc_id}/structures` body:**

| Field | Required | Description |
|---|---|---|
| `section_title` | Yes | e.g. "Chapter 3: The Holy Spirit" |
| `start_page` | Yes | 1-based start page (inclusive) |
| `end_page` | Yes | 1-based end page (inclusive, must be >= start_page) |
| `level` | No | Hierarchy depth: 1 = chapter, 2 = subsection (default: 1) |

---

### Ingest

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest/pdf` | Upload a PDF, DOCX, or TXT file. Pages are stored individually. |
| `POST` | `/ingest/url` | Scrape a web article URL and store its text as a single page. |

**`POST /ingest/pdf` form fields:**

| Field | Required | Description |
|---|---|---|
| `file` | Yes | The document file (PDF, DOCX, or TXT) |
| `title` | Yes | Document title |
| `summary` | Yes | Short summary used for Tier 1 vector routing |
| `author` | No | Author name |
| `date_published` | No | Publication date (YYYY-MM-DD) |
| `focus_area` | No | e.g. "Pauline epistles", "Eschatology" |

**`POST /ingest/url` form fields:**

| Field | Required | Description |
|---|---|---|
| `url` | Yes | The web article URL to scrape |
| `title` | Yes | Document title |
| `summary` | Yes | Short summary used for Tier 1 vector routing |
| `author` | No | Author name |
| `date_published` | No | Publication date (YYYY-MM-DD) |
| `focus_area` | No | e.g. "Pauline epistles", "Eschatology" |

---

### Admin *(UI only, excluded from API schema)*

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/login` | Admin login page |
| `POST` | `/admin/login` | Submit admin password |
| `GET` | `/admin/logout` | Log out of admin session |
| `GET` | `/admin` | Admin dashboard (requires auth) |

---

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Set the following environment variables (or use a `.env` file):

```
OPENAI_API_KEY=...
DATABASE_URL=postgresql://...
```

