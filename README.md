# DocuMentor — RFC Intelligence

A RAG-powered chat assistant for querying IETF RFC documents. Point it at one or more RFC numbers and it fetches the official text, indexes it with FAISS, automatically pulls in cross-referenced RFCs, and answers your questions with section-level citations — all through a single-page dark-themed chat UI.

> Backend: `api.py` (Flask + LangChain + FAISS + Groq)
> Frontend: `index.html` (vanilla HTML/CSS/JS, no build step)

---

## How it works

1. You pick one or more RFC numbers in the sidebar (e.g. `9110`, `8446`).
2. The frontend sends your question + selected RFC numbers + recent chat history to the backend's `/query` endpoint.
3. For each requested RFC, the backend:
   - Downloads the raw text from `rfc-editor.org` (skipped if already cached in memory).
   - Strips headers, footers, and page-break noise.
   - Extracts metadata (title, date, status, `Obsoletes:`, `Updates:`).
   - Scans the text for other RFC numbers it references (cross-references).
   - Splits the cleaned text into overlapping chunks and tags each chunk with its detected section number.
   - Embeds the chunks (`all-MiniLM-L6-v2`) and builds a per-RFC FAISS index.
4. Up to `MAX_CROSS_REFS` cross-referenced RFCs per primary RFC are auto-fetched and indexed the same way.
5. All indexes (primary RFC(s) + auto-loaded cross-refs) are merged into a single combined FAISS store.
6. The combined store is searched with MMR retrieval (`k=8`, `fetch_k=25`) to pull the most relevant, diverse chunks.
7. A prompt (with document metadata, conversation history, and retrieved context) is sent to Groq's `llama-3.1-8b-instant` model, instructed to cite RFC numbers and section numbers and never fabricate protocol details.
8. The answer is returned along with a structured list of sources (RFC number, section, preview snippet, whether it came from a cross-referenced RFC).

## Features

- **Multi-RFC querying** — select and query several RFCs at once.
- **Automatic cross-reference resolution** — if RFC 9110 references RFC 9112, that RFC is fetched and folded into the search index automatically.
- **Section-aware citations** — answers and source chips reference specific section numbers where they can be detected.
- **Conversation memory** — recent Q&A history is fed back into the prompt so follow-up questions work naturally.
- **In-memory caching** — once an RFC is fetched and indexed, it's reused across requests (with manual eviction support).
- **No fabrication policy** — the prompt explicitly forbids inventing protocol fields, timers, bit-widths, or error codes.
- **Quick-load presets** — one-click loading for commonly used RFCs (IPv4, TCP, HTTP/1.1, HTTP/2, HTTP/3, TLS 1.3, DNS) baked into the UI.

## Tech stack

**Backend**
- Python, Flask, Flask-CORS
- LangChain (`langchain-core`, `langchain-text-splitters`, `langchain-community`, `langchain-huggingface`, `langchain-groq`)
- FAISS for vector search
- `sentence-transformers` model `all-MiniLM-L6-v2` for embeddings
- Groq API (`llama-3.1-8b-instant`) for generation

**Frontend**
- Plain HTML/CSS/JS — no framework, no build step
- Talks to the backend over a single `fetch` POST call

## Project structure

```
.
├── api.py        # Flask backend — RAG pipeline, FAISS indexing, Groq calls
└── index.html    # Frontend — chat UI, RFC sidebar, source citations
```

## Getting started

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com)

### Backend

Install the dependencies (there's no `requirements.txt` yet — install directly, or freeze your own once it's working):

```bash
pip install flask flask-cors requests \
    langchain-text-splitters langchain-core langchain-community \
    langchain-huggingface langchain-groq \
    faiss-cpu sentence-transformers
```

Set your Groq API key and start the server:

```bash
export GROQ_API_KEY="your_key_here"   # Windows (PowerShell): $env:GROQ_API_KEY="your_key_here"
python api.py
```

The API starts at `http://localhost:5000`. Check it's alive:

```bash
curl http://localhost:5000/health
```

### Frontend

`index.html` is a static file — open it directly in a browser, or serve it with any static file server.

By default it points at a deployed backend URL set in the `BACKEND_URL` constant near the top of the `<script>` block:

```js
const BACKEND_URL = 'https://documentor-api-hpba.onrender.com';
```

To use your local backend instead, change this to:

```js
const BACKEND_URL = 'http://localhost:5000/query';
```

> Note: the backend's `/query` route is `POST /query`, so `BACKEND_URL` should include that path if you're pointing at a route other than the root.

## API reference

### `GET /health`
Returns service status, the active model, and currently cached RFC numbers.

### `POST /query`
Main RAG endpoint.

**Request body:**
```json
{
  "query": "What are the security considerations?",
  "rfc_numbers": ["9110", "8446"],
  "history": [
    { "role": "user", "content": "previous question" },
    { "role": "assistant", "content": "previous answer" }
  ]
}
```

**Response body:**
```json
{
  "answer": "...",
  "sources": [
    {
      "rfc": "9110",
      "label": "RFC 9110",
      "section": "4.3.2",
      "is_xref": false,
      "preview": "..."
    }
  ],
  "meta": {
    "primary_rfcs": ["9110"],
    "cross_ref_rfcs": ["9112"],
    "rfcs_failed": [],
    "chunks_searched": 8,
    "latency_ms": 1423
  }
}
```

### `GET /cache`
Lists currently cached RFC numbers and the count.

### `DELETE /cache/<rfc_number>`
Evicts a single RFC from the in-memory cache, forcing it to be re-fetched and re-indexed on next use.

## Configuration

These live as constants near the top of `api.py`:

| Constant | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Generation model used via Groq |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers embedding model |
| `MAX_CROSS_REFS` | `4` | Max auto-loaded cross-referenced RFCs per primary RFC |

`GROQ_API_KEY` must be set as an environment variable.

## Limitations

- The RFC cache is in-memory only — it resets whenever the Flask process restarts.
- CORS is fully open (`CORS(app)` with no restrictions) — tighten this before deploying publicly.
- Section detection relies on regex pattern-matching against RFC plain-text formatting, so some chunks may have no detected section.
- No authentication on any route.

## License

This project is licensed under the [MIT License](LICENSE).
