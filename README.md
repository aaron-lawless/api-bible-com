# AI Bible — RAG API

A document ingestion and semantic search API for Christian theological content. Accepts document uploads, chunks and embeds them using OpenAI, and answers natural language questions by retrieving relevant chunks and summarising with GPT-4o.

## Tech Stack

- **Backend**: Python 3.11, FastAPI, SQLAlchemy
- **Database**: PostgreSQL with pgvector
- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Completions**: `gpt-4o`
- **Text extraction**: pdfplumber (PDF), python-docx (DOCX)
- **Deployment**: Railway

---

## Query Caching

Every question passes through three layers before hitting the AI. This minimises API costs and latency for repeated or similar questions.

### Step 1 — Exact Match (zero API calls)

The question is normalised (lowercased, stopwords removed, lemmatized) and hashed with SHA-256. If the same normalised hash exists in `query_cache` from a previous non-cached answer, the stored response and sources are returned immediately.

**Example:**
- Previously asked: *"What does Paul say about grace?"*
- Asked again: *"What does Paul say about grace?"*
- Both normalise to `paul say grace` → same hash → instant return, no API call.

---

### Step 2 — Vector Match (1 embedding call, no completion call)

If no exact hash match is found, the question is embedded and compared via cosine similarity against all previous non-cache-hit rows. If the closest match scores above **0.92**, the existing response and sources are reused.

**Example:**
- Previously asked: *"What does Paul say about grace?"*
- Now asked: *"What did Paul write about God's grace?"*
- Different hash (Step 1 misses) → embedding generated → similarity = 0.95 → response reused, logged as a vector cache hit.

---

### Step 3 — Full RAG (embedding + chunk search + GPT completion)

If no cache hit is found, the full pipeline runs: relevant document chunks are retrieved via vector search, GPT-4o generates an answer grounded in those chunks, and the result is stored for future reuse.

**Example:**
- Asked: *"How does the Gospel of John describe the Holy Spirit?"*
- No hash match, no similar previous question → chunks retrieved, GPT generates answer → stored as a fresh row.

---

### Cache Record

Each query (hit or miss) is recorded in `query_cache` with:

| Field | Description |
|---|---|
| `question_raw` | Original question text |
| `question_normalized` | Normalised form used for hashing |
| `question_hash` | SHA-256 of the normalised question |
| `embedding` | Vector embedding of the normalised question |
| `response` | The answer returned to the user |
| `sources` | Deduplicated list of `{document_id, title, author}` |
| `cache_hit` | Whether this was served from cache |
| `cache_hit_type` | `"exact"`, `"vector"`, or `null` |
| `cache_source_id` | ID of the originating row (for hits) |
| `similarity_score` | Cosine similarity (vector hits only) |
| `token_information` | Embedding and/or completion token counts |
| `session_id` | Client session identifier |

**Example `sources` value:**
```json
[
  {
    "document_id": "a3f1c2d4-8b5e-4f2a-9c1d-7e6f3a2b1c0d",
    "title": "Systematic Theology",
    "author": "Wayne Grudem"
  },
  {
    "document_id": "b7e2d1f3-4a6c-4e8b-8d2e-1f9a3c7b5d2e",
    "title": "The Case for Christ",
    "author": "Lee Strobel"
  }
]
```

---

## API Endpoints

### Search

| Method | Path | Description |
|---|---|---|
| `POST` | `/search` | Ask a question. Returns an answer and sources. Hits cache or calls GPT-4o. |
| `GET` | `/questions` | Paginated list of previously answered questions. |
| `GET` | `/questions/{query_id}` | Retrieve the full answer for a specific question by ID. No API call made. |

**`GET /questions` query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `page` | `1` | Page number |
| `page_size` | `20` | Results per page (max 100) |
| `search` | — | Optional case-insensitive text filter on the question |

---

### Documents

| Method | Path | Description |
|---|---|---|
| `GET` | `/documents` | List all ingested documents (newest first). |
| `GET` | `/documents/{doc_id}` | Get a document and its chunk count. |
| `DELETE` | `/documents/{doc_id}` | Delete a document and all its chunks. |

---

### Ingest

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a PDF, DOCX, or TXT file to chunk, embed, and store. |

**`POST /ingest` form fields:**

| Field | Required | Description |
|---|---|---|
| `file` | Yes | The document file (PDF, DOCX, TXT) |
| `title` | Yes | Document title |
| `author` | No | Author name |
| `isbn` | No | ISBN |
| `date_published` | No | Publication date (YYYY-MM-DD) |
| `description` | No | Short description |
| `source` | No | Source URL or reference |

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
