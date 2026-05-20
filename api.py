"""
DocuMentor — Flask API Backend
Matches notebook behaviour exactly:
  - Auto-fetches cross-referenced RFCs and merges them into the FAISS index
  - Uses the exact RFC_PROMPT from the notebook (section citation rules)
  - Always searches ALL loaded RFCs (primary + cross-refs) together
  - Returns per-chunk sources with RFC number + section for the frontend
"""

import os, re, time, tempfile, requests, traceback as tb_module
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GROQ_MODEL     = "llama-3.1-8b-instant"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RFC_TEXT_URL   = "https://www.rfc-editor.org/rfc/rfc{}.txt"
MAX_CROSS_REFS = 4

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

app = Flask(__name__)
CORS(app)

_rfc_cache: dict = {}

print(f"Loading embedding model '{EMBED_MODEL}' ...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding model ready.\n")

# ── Exact RFC_PROMPT from notebook ────────────────────────────────────────────
RFC_PROMPT = """You are an expert RFC (Request for Comments) analyst.
Answer the question using ONLY the context sections provided below.
If multiple RFCs are loaded, clearly state which RFC each part of your answer comes from.

{metadata_block}

{chat_history}

CONTEXT:
{context}

RULES:
- Cite section numbers when available, e.g. "Section 4.2 states..."
- Write all RFC references as "RFC XXXX" format.
- Always look for the answer in the primary RFC first. Only if not present, use cross-linked RFCs.
- Be concise but thorough. Use bullet points or headers where helpful.
- Never fabricate protocol fields, timer values, bit-widths, or error codes.
- Use the conversation history to understand follow-up questions.

Question: {question}

Answer:"""


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_rfc_text(raw: str) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    t = re.sub(r"^RFC\s+\d+\s+.+\n?",                          "", t, flags=re.M)
    t = re.sub(r"\[Page\s+\d+\]",                              "", t, flags=re.I)
    t = re.sub(r"^[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\s{3,}.+\n", "", t, flags=re.M)
    t = re.sub(r"[ \t]+$",                                     "", t, flags=re.M)
    t = re.sub(r"^[-=_]{3,}\s*$",                              "", t, flags=re.M)
    t = re.sub(r"\n{4,}",                                      "\n\n\n", t)
    return t.strip()


def extract_metadata(text: str) -> dict:
    def grab(pattern, flags=re.I | re.M):
        m = re.search(pattern, text, flags)
        if not m:
            return None
        value = re.split(r"  +", m.group(1))[0]
        return re.sub(r"\s+", " ", value).strip() or None
    return {
        "rfcNumber": grab(r"^(?:Request for Comments|RFC)[:\s]+(\d+)"),
        "title":     grab(r"^(?!\s*\w+:)([A-Z][^\n]{10,80})\n\s*\n"),
        "date":      grab(r"((?:January|February|March|April|May|June|July|"
                          r"August|September|October|November|December)\s+\d{4})"),
        "status":    grab(r"^(?:Status|Category)\s*:\s*([^\n]+)"),
        "obsoletes": grab(r"^Obsoletes:\s*([^\n]+)"),
        "updates":   grab(r"^Updates:\s*([^\n]+)"),
    }


def extract_rfc_links(text: str) -> list:
    ids = set()
    for pat in [r"\bRFC\s*[-–]?\s*(\d{3,5})\b", r"\[RFC(\d{3,5})\]"]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            ids.add(m.group(1))
    return sorted(ids, key=int)


def extract_section(chunk: str) -> Optional[str]:
    """
    Scan every line for RFC section headings like:
      4.3.2  Title        (2+ spaces after number)
      1.  Introduction    (dot then 2+ spaces)
    Returns first match or None.
    """
    for line in chunk.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Z]?\d+(?:\.\d+)*)\.?\s{2,}\S', line)
        if m:
            sec = m.group(1)
            # Sanity: ignore things like "1234  " that are just numbers with no dot
            if len(sec) <= 10:
                return sec
    return None


def split_into_documents(text: str, source_label: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        length_function=len,
    )
    docs = []
    for i, chunk in enumerate(splitter.split_text(text)):
        section = extract_section(chunk)   # None if not found — never "chunk-N"
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source":  source_label,
                "section": section,
                "chunk_index": i,
            },
        ))
    return docs


def fetch_rfc_text(rfc_number: str) -> Optional[str]:
    try:
        print(f"  Fetching RFC {rfc_number} ...", flush=True)
        r = requests.get(RFC_TEXT_URL.format(rfc_number), timeout=20,
                         headers={"User-Agent": "DocuMentor/1.0"})
        r.raise_for_status()
        return clean_rfc_text(r.text)
    except Exception as e:
        print(f"  Error fetching RFC {rfc_number}: {e}")
        return None


def clone_vs(vs: FAISS) -> FAISS:
    with tempfile.TemporaryDirectory() as tmp:
        vs.save_local(tmp)
        return FAISS.load_local(tmp, embeddings, allow_dangerous_deserialization=True)


# ── Cache ─────────────────────────────────────────────────────────────────────

def ensure_rfc_cached(rfc_number: str) -> bool:
    if rfc_number in _rfc_cache:
        return True
    text = fetch_rfc_text(rfc_number)
    if text is None:
        return False
    meta       = extract_metadata(text)
    cross_refs = extract_rfc_links(text)
    docs       = split_into_documents(text, f"RFC {rfc_number}")
    vs         = FAISS.from_documents(docs, embeddings)
    _rfc_cache[rfc_number] = {
        "vs": vs, "meta": meta, "cross_refs": cross_refs, "chunks": len(docs)
    }
    print(f"  RFC {rfc_number} indexed: {len(docs)} chunks, {len(cross_refs)} cross-refs.")
    return True


def build_combined_store(primary_numbers: list) -> tuple:
    loaded, failed, auto_loaded = [], [], []

    # Index primaries
    for num in primary_numbers:
        if ensure_rfc_cached(num):
            loaded.append(num)
        else:
            failed.append(num)

    if not loaded:
        return None, loaded, failed, auto_loaded

    # Auto-load cross-refs
    seen = set(loaded)
    cross_queue = []
    for num in loaded:
        for ref in _rfc_cache[num]["cross_refs"]:
            if ref not in seen:
                cross_queue.append(ref)
                seen.add(ref)

    for ref in cross_queue[:MAX_CROSS_REFS * len(loaded)]:
        if ensure_rfc_cached(ref):
            auto_loaded.append(ref)

    # Merge ALL into one FAISS store
    all_loaded  = loaded + auto_loaded
    combined_vs = clone_vs(_rfc_cache[all_loaded[0]]["vs"])
    for num in all_loaded[1:]:
        combined_vs.merge_from(_rfc_cache[num]["vs"])

    print(f"  Combined index: {loaded} primary + {auto_loaded} cross-refs")
    return combined_vs, loaded, failed, auto_loaded


# ── RAG ───────────────────────────────────────────────────────────────────────

def run_rag(query: str, combined_vs: FAISS,
            primary_rfcs: list, all_rfcs: list,
            history: list) -> tuple:
    """
    Always searches the FULL combined index (primary + cross-refs).
    This ensures cross-referenced content is always available.
    The prompt instructs the LLM to prefer primary RFC content.
    """

    # Search full index — k=8 to get good coverage across RFCs
    retrieved_docs = combined_vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25, "lambda_mult": 0.7},
    ).invoke(query)

    print(f"  Retrieved {len(retrieved_docs)} chunks from full index "
          f"({len(all_rfcs)} RFCs)")

    # Build metadata block
    primary_meta = _rfc_cache.get(primary_rfcs[0], {}).get("meta", {}) if primary_rfcs else {}
    meta_lines = [
        "DOCUMENT INFO:",
        f"  Primary RFC : RFC {', '.join(primary_rfcs)} — {primary_meta.get('title', '')}",
        f"  All loaded  : {', '.join(f'RFC {n}' for n in all_rfcs)}",
    ]
    if primary_meta.get("obsoletes"):
        meta_lines.append(f"  Obsoletes   : {primary_meta['obsoletes']}")
    if primary_meta.get("updates"):
        meta_lines.append(f"  Updates     : {primary_meta['updates']}")

    # Build history block
    history_block = ""
    if history:
        lines = ["CONVERSATION HISTORY (most recent last):"]
        for i, turn in enumerate(history[-5:], 1):
            role    = turn.get("role", "")
            content = (turn.get("content") or "")[:500]
            if role == "user":
                lines.append(f"  Q{i}: {content}")
            elif role == "assistant":
                lines.append(f"  A{i}: {content}")
        history_block = "\n".join(lines)

    # Build context — include source label and section so LLM can cite them
    def chunk_header(d):
        src = d.metadata.get("source", "?")
        sec = d.metadata.get("section")
        return f"[{src}{', Section ' + sec if sec else ''}]"

    context_text = "\n\n---\n\n".join(
        f"{chunk_header(d)}\n{d.page_content}"
        for d in retrieved_docs
    )

    prompt = PromptTemplate(
        template=RFC_PROMPT
            .replace("{metadata_block}", "\n".join(meta_lines))
            .replace("{chat_history}", history_block),
        input_variables=["context", "question"],
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=1024,
    )

    chain = (
        {"context": lambda _: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    return answer, retrieved_docs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    tb = tb_module.format_exc()
    print("\n=== UNHANDLED EXCEPTION ===\n" + tb + "===========================\n")
    return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "model":       GROQ_MODEL,
        "cached_rfcs": list(_rfc_cache.keys()),
    })


@app.route("/query", methods=["POST"])
def query_route():
    data        = request.get_json(force=True)
    user_query  = (data.get("query") or "").strip()
    rfc_numbers = data.get("rfc_numbers") or []
    history     = data.get("history") or []

    if not user_query:
        return jsonify({"error": "query is required"}), 400
    if not rfc_numbers:
        return jsonify({"error": "rfc_numbers is required"}), 400

    rfc_numbers = list(dict.fromkeys(
        re.sub(r"\D", "", str(n)) for n in rfc_numbers
    ))
    rfc_numbers = [n for n in rfc_numbers if n]

    print(f"\n[/query] '{user_query[:70]}' | primaries: {rfc_numbers}")

    t0 = time.perf_counter()
    combined_vs, loaded, failed, auto_loaded = build_combined_store(rfc_numbers)

    if combined_vs is None:
        return jsonify({"error": f"Could not load RFC(s): {rfc_numbers}"}), 502

    all_rfcs = loaded + auto_loaded

    try:
        answer, source_docs = run_rag(user_query, combined_vs, loaded, all_rfcs, history)
    except Exception as e:
        print(f"  RAG error: {e}\n{tb_module.format_exc()}")
        return jsonify({"error": f"RAG error: {str(e)}"}), 500

    ms = round((time.perf_counter() - t0) * 1000)

    # Build rich source list — skip duplicates, never show None section
    seen_keys = set()
    sources   = []
    for d in source_docs:
        rfc_label = d.metadata.get("source", "")
        section   = d.metadata.get("section")   # already None if not found
        key       = f"{rfc_label}|{section}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rfc_num   = rfc_label.replace("RFC ", "").strip()
        is_xref   = rfc_num not in loaded
        sources.append({
            "rfc":     rfc_num,
            "label":   rfc_label,
            "section": section,          # None → frontend shows no §
            "is_xref": is_xref,
            "preview": d.page_content[:150].replace("\n", " ").strip() + "…",
        })

    src_summary = [s["label"] + (" §" + s["section"] if s["section"] else "") for s in sources]
    print(f"  Done {ms}ms | sources: {src_summary}")

    return jsonify({
        "answer":  answer,
        "sources": sources,
        "meta": {
            "primary_rfcs":    loaded,
            "cross_ref_rfcs":  auto_loaded,
            "rfcs_failed":     failed,
            "chunks_searched": len(source_docs),
            "latency_ms":      ms,
        },
    })


@app.route("/cache", methods=["GET"])
def cache_status():
    return jsonify({"cached": list(_rfc_cache.keys()), "count": len(_rfc_cache)})


@app.route("/cache/<rfc_number>", methods=["DELETE"])
def evict_cache(rfc_number):
    num = re.sub(r"\D", "", rfc_number)
    if num in _rfc_cache:
        del _rfc_cache[num]
        return jsonify({"evicted": num})
    return jsonify({"error": f"RFC {num} not in cache"}), 404


if __name__ == "__main__":
    print("DocuMentor API → http://localhost:5000")
    print("Health check  → http://localhost:5000/health\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
