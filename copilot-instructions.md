GitHub Copilot prompt — paste into Copilot Chat or save as .github/copilot-instructions.md

You are building a production-ready Flask RAG (Retrieval-Augmented Generation) application. Build the complete application according to the spec below. Write all code, migrations, and configuration files needed to run this on Railway.

---

## Project overview

A document ingestion and semantic search API that:
- Accepts large document uploads (PDF, DOCX, TXT) with metadata
- Chunks and embeds documents using OpenAI text-embedding-3-small
- Stores chunks and embeddings in PostgreSQL with pgvector on Railway
- Answers natural language questions by retrieving relevant chunks and summarising with gpt-4o
- Exposes a clean Flask REST API with a minimal search UI

---

## Tech stack

- **Backend**: Python 3.11, Flask, SQLAlchemy (ORM), Alembic (migrations)
- **Database**: PostgreSQL 15 on Railway with pgvector extension
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Completions**: gpt-4o
- **Text extraction**: pdfplumber (PDF), python-docx (DOCX)
- **Chunking**: langchain-text-splitters RecursiveCharacterTextSplitter
- **Env management**: python-dotenv

---

## Database schema

### documents table
| column | type | notes |
|---|---|---|
| id | UUID primary key | gen_random_uuid() |
| title | TEXT NOT NULL | |
| author | TEXT | |
| isbn | TEXT | |
| date_published | DATE | |
| description | TEXT | |
| file_path | TEXT | original upload path |
| created_at | TIMESTAMPTZ | default now() |

### chunks table
| column | type | notes |
|---|---|---|
| id | UUID primary key | |
| document_id | UUID FK → documents.id ON DELETE CASCADE | |
| chunk_index | INTEGER | order within document |
| content | TEXT NOT NULL | raw chunk text |
| embedding | vector(1536) | pgvector column |
| created_at | TIMESTAMPTZ | default now() |

Create an IVFFLAT index on chunks.embedding for cosine distance:
```sql
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

---

## File structure to generate

```
/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Config from env vars
│   ├── models.py            # SQLAlchemy models for documents + chunks
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── ingest.py        # POST /ingest
│   │   ├── search.py        # POST /search
│   │   └── documents.py     # GET /documents, GET /documents/, DELETE /documents/
│   ├── services/
│   │   ├── extractor.py     # Text extraction (PDF, DOCX, TXT)
│   │   ├── chunker.py       # RecursiveCharacterTextSplitter wrapper
│   │   ├── embedder.py      # OpenAI embedding calls with retry + batching
│   │   └── search.py        # pgvector cosine search + GPT-4o completion
│   └── templates/
│       └── index.html       # Minimal search UI (single page, vanilla JS)
├── migrations/              # Alembic migrations
├── tests/
│   ├── test_ingest.py
│   └── test_search.py
├── .env.example
├── requirements.txt
├── railway.toml
└── run.py
```

---

## API specification

### POST /ingest
- Accepts multipart/form-data
- Fields: `file` (required), `title` (required), `author`, `isbn`, `date_published` (YYYY-MM-DD), `description`
- Flow: save file → extract text → chunk (500 tokens, 50 overlap) → batch embed (20 chunks per API call) → insert document row → bulk insert chunk rows
- Returns: `{ "document_id": "", "chunk_count": N, "title": "..." }`
- Error handling: unsupported file type (400), OpenAI API failure (502), DB error (500)

### POST /search
- Body: `{ "query": "...", "top_k": 10, "document_ids": [""] }` (document_ids optional filter)
- Flow: embed query → cosine similarity search in pgvector → build context from top-K chunks → GPT-4o completion
- Returns: `{ "answer": "...", "sources": [{ "document_id", "title", "author", "chunk_index", "content" }] }`

### GET /documents
- Returns list of all documents with metadata (no chunks/embeddings)

### GET /documents/
- Returns document metadata + chunk count

### DELETE /documents/
- Cascades to chunks via FK

---

## Service implementation details

### extractor.py
- Use pdfplumber for PDF (iterate pages, join text)
- Use python-docx for DOCX (iterate paragraphs)
- Plain read() for TXT with UTF-8 fallback to latin-1
- Raise ValueError for unsupported extensions

### chunker.py
- Use RecursiveCharacterTextSplitter with chunk_size=500, chunk_overlap=50, length_function=tiktoken cl100k_base token count
- Return list of dicts: `{ "content": str, "chunk_index": int }`

### embedder.py
- Batch chunks in groups of 20
- Call openai.embeddings.create(model="text-embedding-3-small", input=[...])
- Exponential backoff on RateLimitError (tenacity, max 3 retries)
- Return list of 1536-dim float lists in same order as input

### search.py (service)
- pgvector query: `SELECT c.*, d.title, d.author FROM chunks c JOIN documents d ON c.document_id = d.id ORDER BY c.embedding <=> :query_vec LIMIT :top_k`
- If document_ids filter provided, add `WHERE c.document_id = ANY(:doc_ids)`
- GPT-4o system prompt: "You are a research assistant. Answer the question using only the provided document excerpts. Cite the source title and chunk index for each claim."
- GPT-4o user message: include query + formatted chunks with source labels
- max_tokens: 1024, temperature: 0

---

## Configuration (.env.example)

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
OPENAI_API_KEY=sk-...
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=52428800
FLASK_ENV=development
SECRET_KEY=change-me
```

---

## railway.toml

```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "flask db upgrade && gunicorn run:app --workers 2 --bind 0.0.0.0:$PORT"
healthcheckPath = "/health"
```

---

## Requirements

Include exact pinned versions for:
flask, sqlalchemy, alembic, psycopg2-binary, pgvector, openai, pdfplumber, python-docx, tiktoken, langchain-text-splitters, tenacity, python-dotenv, gunicorn

---

## Additional requirements

- All routes must return JSON errors with `{ "error": "..." }` and appropriate HTTP status codes
- Log all ingestion and search events with Python logging (not print)
- The index.html template should have a file upload form and a search box that calls the API via fetch, displaying the answer and sources
- Write pytest tests for ingest (mock OpenAI) and search (mock OpenAI + pgvector query)
- Add a GET /health endpoint returning `{ "status": "ok" }`

Generate all files completely. Do not truncate any file. Start with requirements.txt, then models.py, then services, then routes, then templates, then tests, then config files.