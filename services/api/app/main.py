# import uuid
# import os
# from time import perf_counter
# from contextlib import asynccontextmanager
# from typing import Any, Dict, List

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from sqlalchemy import select, delete

# from app.db import SessionLocal
# from app.models import Document, Chunk
# from app.chunking import chunk_text
# from app.search import rank_chunks
# from app.settings import get_settings
# from app.embeddings import EmbeddingProvider
# from app.vectorstore import VectorStore
# from app.lexical import lexical_search
# from app.retrieval import rrf_fuse
# from app.mmr import mmr_select
# from app.mmr_utils import count_redundant_items
# from app.rerank import Reranker
# from app.llm.provider import get_llm_provider
# from app.citations import build_sources_block, extract_citation_numbers


# # ---------------------------------------------------------------------
# # Settings / Globals
# # ---------------------------------------------------------------------

# MAX_RERANK_CANDIDATES = 50  # production cap (latency + memory control)

# settings = get_settings()
# embedder = EmbeddingProvider(settings.embedding_model)
# vstore = VectorStore(settings.qdrant_url, settings.qdrant_collection)

# reranker = None
# if os.environ.get("ENABLE_RERANK", "false").lower() == "true":
#     reranker = Reranker(os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Ensure Qdrant collection exists on startup
#     vstore.ensure_collection(vector_size=embedder.dim())
#     yield


# app = FastAPI(lifespan=lifespan)

# # ---------------------------------------------------------------------
# # Models
# # ---------------------------------------------------------------------

# class IngestRequest(BaseModel):
#     title: str
#     content: str


# class VectorSearchRequest(BaseModel):
#     query: str
#     top_k: int = 5


# class SearchRequest(BaseModel):
#     query: str
#     top_k: int = 5


# class RetrieveRequest(BaseModel):
#     query: str
#     top_k: int = Field(5, ge=1, le=50)

#     lexical_k: int = Field(20, ge=1, le=200)
#     vector_k: int = Field(20, ge=1, le=200)

#     # candidates_k is the pool size after fusion; rerank (if enabled) will be capped to MAX_RERANK_CANDIDATES
#     candidates_k: int = Field(50, ge=5, le=200)

#     use_mmr: bool = True
#     mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)          # production default
#     mmr_dup_threshold: float = Field(0.95, ge=0.0, le=1.0)   # cosine similarity threshold

#     use_rerank: bool = False

#     # If True, return {"results": [...], "meta": {...}}; else return the list only (backward compatible)
#     include_meta: bool = False


# # ---------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------

# def fetch_chunks(db, ids: List[str]) -> Dict[str, Dict[str, Any]]:
#     rows = db.execute(
#         select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content)
#         .where(Chunk.id.in_(ids))
#     ).all()
#     return {
#         r[0]: {"chunk_id": r[0], "document_id": r[1], "chunk_index": r[2], "content": r[3]}
#         for r in rows
#     }


# def format_result(it: Dict[str, Any], selected_by: str) -> Dict[str, Any]:
#     return {
#         "chunk_id": it["chunk_id"],
#         "document_id": it["document_id"],
#         "chunk_index": it["chunk_index"],
#         "content_preview": it["content"][:200],
#         "final_score": it["final_score"],

#         # explainability/debug
#         "selected_by": selected_by,

#         # hybrid debugging (already useful in interviews)
#         "rrf_score": it.get("rrf_score"),
#         "lexical_rank": it.get("lexical_rank"),
#         "vector_rank": it.get("vector_rank"),
#         "lexical_score": it.get("lexical_score"),
#         "vector_score": it.get("vector_score"),

#         # rerank debugging
#         "rerank_score": it.get("rerank_score"),
#     }


# # ---------------------------------------------------------------------
# # Routes
# # ---------------------------------------------------------------------

# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/dev/reset")
# def dev_reset():
#     if os.environ.get("ENV") != "dev":
#         raise HTTPException(status_code=404, detail="Not found")

#     with SessionLocal() as db:
#         db.execute(delete(Chunk))
#         db.execute(delete(Document))
#         db.commit()

#     return {"status": "reset ok"}


# @app.post("/documents")
# def create_document(req: IngestRequest):
#     if not req.title.strip():
#         raise HTTPException(status_code=400, detail="title cannot be empty")
#     if not req.content.strip():
#         raise HTTPException(status_code=400, detail="content cannot be empty")

#     doc_id = str(uuid.uuid4())
#     chunks = chunk_text(req.content)

#     with SessionLocal() as db:
#         db.add(Document(id=doc_id, title=req.title, content=req.content))
#         db.flush()  # ensures doc exists for FK

#         chunk_rows = []  # (chunk_id, chunk_index, text)
#         for i, chunk in enumerate(chunks):
#             chunk_id = str(uuid.uuid4())
#             db.add(
#                 Chunk(
#                     id=chunk_id,
#                     document_id=doc_id,
#                     chunk_index=i,
#                     content=chunk,
#                 )
#             )
#             chunk_rows.append((chunk_id, i, chunk))

#         texts = [c[2] for c in chunk_rows]
#         vectors = embedder.embed(texts)

#         points = []
#         for (chunk_id, idx, _content), vec in zip(chunk_rows, vectors):
#             points.append(
#                 {
#                     "id": chunk_id,
#                     "vector": vec,
#                     "payload": {
#                         "document_id": doc_id,
#                         "chunk_index": idx,
#                     },
#                 }
#             )
#         vstore.upsert_chunks(points)

#         db.commit()

#     return {"id": doc_id, "chunks_created": len(chunks)}


# @app.post("/search_vector")
# def search_vector(req: VectorSearchRequest):
#     qvec = embedder.embed([req.query])[0]
#     hits = vstore.search(qvec, limit=req.top_k)

#     hit_ids = [str(h.id) for h in hits]
#     with SessionLocal() as db:
#         rows = db.execute(
#             select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content)
#             .where(Chunk.id.in_(hit_ids))
#         ).all()

#     row_map = {
#         r[0]: {
#             "chunk_id": r[0],
#             "document_id": r[1],
#             "chunk_index": r[2],
#             "content_preview": r[3][:200],
#         }
#         for r in rows
#     }

#     results = []
#     for h in hits:
#         cid = str(h.id)
#         if cid in row_map:
#             results.append({"score": float(h.score), **row_map[cid]})

#     return results


# @app.get("/documents")
# def list_documents():
#     with SessionLocal() as db:
#         rows = db.execute(select(Document.id, Document.title)).all()
#     return [{"id": r[0], "title": r[1]} for r in rows]


# @app.get("/documents/{doc_id}/chunks")
# def list_chunks(doc_id: str):
#     with SessionLocal() as db:
#         rows = db.execute(
#             select(Chunk.id, Chunk.chunk_index, Chunk.content)
#             .where(Chunk.document_id == doc_id)
#             .order_by(Chunk.chunk_index)
#         ).all()
#     return [{"id": r[0], "chunk_index": r[1], "content": r[2]} for r in rows]


# @app.post("/search")
# def search(req: SearchRequest):
#     with SessionLocal() as db:
#         rows = db.execute(
#             select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content)
#         ).all()

#     ranked = rank_chunks(req.query, rows, top_k=req.top_k)
#     return [
#         {
#             "score": s,
#             "chunk_id": chunk_id,
#             "document_id": doc_id,
#             "chunk_index": idx,
#             "content_preview": content[:200],
#         }
#         for (s, chunk_id, doc_id, idx, content) in ranked
#     ]


# @app.post("/retrieve")
# def retrieve(req: RetrieveRequest):
#     t0 = perf_counter()
#     timings: Dict[str, float] = {}

#     # --- Lexical
#     t = perf_counter()
#     with SessionLocal() as db:
#         lex = lexical_search(db, req.query, top_k=req.lexical_k)
#     timings["lexical_ms"] = (perf_counter() - t) * 1000.0

#     # --- Vector
#     t = perf_counter()
#     qvec = embedder.embed([req.query])[0]
#     vhits = vstore.search(qvec, limit=req.vector_k)
#     vec = [{"chunk_id": str(h.id), "score": float(h.score)} for h in vhits]
#     timings["vector_ms"] = (perf_counter() - t) * 1000.0

#     # --- Fuse (RRF)
#     t = perf_counter()
#     fused = rrf_fuse(lex, vec, k=60)
#     fused = fused[: req.candidates_k]
#     timings["fuse_ms"] = (perf_counter() - t) * 1000.0

#     # --- Fetch full chunk records
#     t = perf_counter()
#     ids = [x["chunk_id"] for x in fused]
#     with SessionLocal() as db:
#         chunk_map = fetch_chunks(db, ids)
#     candidates: List[Dict[str, Any]] = []
#     for x in fused:
#         cid = x["chunk_id"]
#         if cid in chunk_map:
#             candidates.append({**x, **chunk_map[cid]})
#     timings["fetch_ms"] = (perf_counter() - t) * 1000.0

#     # --- Rerank (optional, capped)
#     t = perf_counter()
#     rerank_cap_used = None

#     if req.use_rerank and reranker is not None:
#         # cap candidates for rerank (professional/production)
#         cap = min(req.candidates_k, MAX_RERANK_CANDIDATES)
#         rerank_cap_used = cap
#         candidates_for_rerank = candidates[:cap]

#         candidates_for_rerank = reranker.rerank(req.query, candidates_for_rerank)
#         for it in candidates_for_rerank:
#             it["final_score"] = it["rerank_score"]

#         candidates = candidates_for_rerank
#     else:
#         for it in candidates:
#             it["final_score"] = it.get("rrf_score", 0.0)

#     timings["rerank_ms"] = (perf_counter() - t) * 1000.0

#     # Baseline top-k (score-only) for redundancy comparison
#     baseline = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[: req.top_k]

#     # --- MMR (optional)
#     t = perf_counter()
#     dropped_duplicates = None
#     baseline_redundant = None
#     selected_redundant = None
#     vectors_found = None
#     vectors_missing = None
#     selected_by = "score"

#     if req.use_mmr:
#         selected_by = "mmr"

#         # get vectors for candidates (needed for MMR + redundancy metric)
#         want = [it["chunk_id"] for it in candidates]
#         vecs = vstore.get_vectors(want)

#         # vector coverage (proves redundancy metrics aren’t lying)
#         found = sum(1 for cid in want if cid in vecs and vecs[cid] is not None)
#         missing = len(want) - found
#         vectors_found = found
#         vectors_missing = missing

#         selected = mmr_select(
#             candidates=candidates,
#             vectors=vecs,
#             top_k=req.top_k,
#             lam=req.mmr_lambda,
#         )

#         # redundancy comparison: how many "near duplicates" did MMR avoid?
#         baseline_redundant = count_redundant_items(baseline, vecs, threshold=req.mmr_dup_threshold)
#         selected_redundant = count_redundant_items(selected, vecs, threshold=req.mmr_dup_threshold)
#         dropped_duplicates = max(0, baseline_redundant - selected_redundant)

#         out = selected
#     else:
#         out = baseline

#     timings["mmr_ms"] = (perf_counter() - t) * 1000.0
#     timings["total_ms"] = (perf_counter() - t0) * 1000.0

#     results = [format_result(it, selected_by=selected_by) for it in out]

#     meta = {
#         "use_mmr": req.use_mmr,
#         "mmr_lambda": req.mmr_lambda if req.use_mmr else None,
#         "mmr_dup_threshold": req.mmr_dup_threshold if req.use_mmr else None,
#         "dropped_duplicates": dropped_duplicates if req.use_mmr else None,
#         "use_rerank": bool(req.use_rerank and reranker is not None),
#         "rerank_candidates_cap": rerank_cap_used,
#         "timings_ms": {k: round(v, 2) for k, v in timings.items()},
#     }

#     # NEW: baseline vs selected redundancy + vector coverage
#     meta.update({
#         "baseline_redundant": baseline_redundant if req.use_mmr else None,
#         "selected_redundant": selected_redundant if req.use_mmr else None,
#         "vectors_found": vectors_found if req.use_mmr else None,
#         "vectors_missing": vectors_missing if req.use_mmr else None,
#     })

#     if req.include_meta:
#         return {"results": results, "meta": meta}

#     return results


# class AnswerRequest(BaseModel):
#     query: str
#     top_k: int = Field(5, ge=1, le=20)

#     # reuse your retrieval knobs
#     lexical_k: int = Field(20, ge=1, le=200)
#     vector_k: int = Field(20, ge=1, le=200)
#     candidates_k: int = Field(50, ge=5, le=200)

#     use_mmr: bool = True
#     mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
#     mmr_dup_threshold: float = Field(0.95, ge=0.0, le=1.0)

#     use_rerank: bool = False

#     include_meta: bool = False  # if true, echo retrieval meta too


# @app.post("/answer")
# def answer(req: AnswerRequest):
#     # 1) call internal retrieve code path directly (same process, no HTTP hop)
#     retrieve_req = RetrieveRequest(
#         query=req.query,
#         top_k=req.top_k,
#         lexical_k=req.lexical_k,
#         vector_k=req.vector_k,
#         candidates_k=req.candidates_k,
#         use_mmr=req.use_mmr,
#         mmr_lambda=req.mmr_lambda,
#         mmr_dup_threshold=req.mmr_dup_threshold,
#         use_rerank=req.use_rerank,
#         include_meta=True,  # always get meta internally; decide what to return later
#     )
#     payload = retrieve(retrieve_req)  # your existing handler returns {"results":..., "meta":...} when include_meta=True
#     results = payload["results"]
#     meta = payload["meta"]

#     # 2) build prompt with numbered sources
#     sources_text, idx_map = build_sources_block(results)

#     system = (
#         "You are a production RAG assistant.\n"
#         "Rules:\n"
#         "- Use ONLY the SOURCES provided.\n"
#         "- If the SOURCES do not contain enough info, say you don't know.\n"
#         "- Every paragraph must include at least one citation like [1].\n"
#         "- Do not cite numbers that do not exist in SOURCES.\n"
#         "- Be concise.\n"
#     )

#     user = f"{sources_text}\n\nUSER QUESTION:\n{req.query}\n\nAnswer with citations."

#     # 3) generate
#     try:
#         llm = get_llm_provider()
#     except RuntimeError as e:
#         raise HTTPException(status_code=503, detail=str(e))

#     out = llm.generate(system=system, user=user).text

#     # 4) map citations back to chunk metadata
#     cited_nums = extract_citation_numbers(out)
#     citations = []
#     for n in cited_nums:
#         if n in idx_map:
#             r = idx_map[n]
#             citations.append({
#                 "source_index": n,
#                 "chunk_id": r.get("chunk_id"),
#                 "document_id": r.get("document_id"),
#                 "chunk_index": r.get("chunk_index"),
#             })

#     resp = {
#         "answer": out,
#         "citations": citations,
#     }
#     if req.include_meta:
#         resp["retrieval_meta"] = meta
#     return resp

import uuid
import os
import re
from time import perf_counter
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy import select, delete

from app.chunking import chunk_text, chunk_pages
from app.parsers.pdf import extract_pages_from_pdf_bytes, PdfExtractError
from app.db import SessionLocal
from app.models import Document, Chunk
from app.chunking import chunk_text
from app.search import rank_chunks
from app.settings import get_settings
from app.embeddings import EmbeddingProvider
from app.vectorstore import VectorStore
from app.lexical import lexical_search
from app.retrieval import rrf_fuse
from app.mmr import mmr_select
from app.mmr_utils import count_redundant_items
from app.rerank import Reranker
from app.llm.provider import get_llm_provider
from app.citations import build_sources_block, extract_citation_numbers
from app.answer_validation import validate_grounded_answer


# ---------------------------------------------------------------------
# Settings / Globals
# ---------------------------------------------------------------------

REFUSAL_PREFIX = "I don't know based on the provided sources."

MAX_RERANK_CANDIDATES = 50  # production cap (latency + memory control)

settings = get_settings()
embedder = EmbeddingProvider(settings.embedding_model)
vstore = VectorStore(settings.qdrant_url, settings.qdrant_collection)

reranker = None
if os.environ.get("ENABLE_RERANK", "false").lower() == "true":
    reranker = Reranker(os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

# Be defensive: allow the API to run even if Chunk.meta isn't present yet.
HAS_CHUNK_META = hasattr(Chunk, "meta")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure Qdrant collection exists on startup
    vstore.ensure_collection(vector_size=embedder.dim())
    yield


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------


class IngestRequest(BaseModel):
    title: str
    content: str


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=50)

    lexical_k: int = Field(20, ge=1, le=200)
    vector_k: int = Field(20, ge=1, le=200)

    # candidates_k is the pool size after fusion; rerank (if enabled) will be capped to MAX_RERANK_CANDIDATES
    candidates_k: int = Field(50, ge=5, le=200)

    use_mmr: bool = True
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)  # production default
    mmr_dup_threshold: float = Field(0.95, ge=0.0, le=1.0)  # cosine similarity threshold

    use_rerank: bool = False

    # If True, return {"results": [...], "meta": {...}}; else return the list only (backward compatible)
    include_meta: bool = False

    doc_ids: Optional[List[str]] = None


class AnswerRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)

    # reuse your retrieval knobs
    lexical_k: int = Field(20, ge=1, le=200)
    vector_k: int = Field(20, ge=1, le=200)
    candidates_k: int = Field(50, ge=5, le=200)

    use_mmr: Optional[bool] = None
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    mmr_dup_threshold: float = Field(0.95, ge=0.0, le=1.0)

    use_rerank: Optional[bool] = None

    # /answer-specific controls
    max_source_chars: int = Field(1200, ge=200, le=4000)
    include_sources: bool = False

    include_meta: bool = False  # if true, echo retrieval meta too

    doc_ids: Optional[List[str]] = None

    strict_refusal: bool = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _is_refusal(text: str) -> bool:
    return (text or "").strip().startswith(REFUSAL_PREFIX)

def _normalize_chunks(chunks: Any) -> List[Dict[str, Any]]:
    """
    Make ingestion resilient:
    - If chunk_text() returns List[str], convert to [{"text": str, "section": "General"}]
    - If it returns List[dict], require a "text" field; default section if missing.
    """
    out: List[Dict[str, Any]] = []
    if not chunks:
        return out

    for ch in chunks:
        if isinstance(ch, str):
            txt = ch
            sec = "General"
        elif isinstance(ch, dict):
            # Accept common keys defensively
            txt = ch.get("text") or ch.get("content") or ""
            sec = ch.get("section") or "General"
        else:
            continue

        txt = (txt or "").strip()
        if not txt:
            continue

        out.append({"text": txt, "section": sec})

    return out


def fetch_chunks(db, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not ids:
        return {}

    if HAS_CHUNK_META:
        rows = db.execute(
            select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content, Chunk.meta).where(
                Chunk.id.in_(ids)
            )
        ).all()
        return {
            r[0]: {
                "chunk_id": r[0],
                "document_id": r[1],
                "chunk_index": r[2],
                "content": r[3],
                "meta": r[4] or {},
            }
            for r in rows
        }

    # Fallback if meta column isn't present yet.
    rows = db.execute(
        select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content).where(Chunk.id.in_(ids))
    ).all()
    return {
        r[0]: {"chunk_id": r[0], "document_id": r[1], "chunk_index": r[2], "content": r[3], "meta": {}}
        for r in rows
    }


def format_result(it: Dict[str, Any], selected_by: str) -> Dict[str, Any]:
    return {
        "chunk_id": it["chunk_id"],
        "document_id": it["document_id"],
        "chunk_index": it["chunk_index"],
        "content_preview": (it.get("content") or "")[:200],
        "final_score": it["final_score"],
        "meta": it.get("meta") or {},

        # explainability/debug
        "selected_by": selected_by,

        # hybrid debugging (already useful in interviews)
        "rrf_score": it.get("rrf_score"),
        "lexical_rank": it.get("lexical_rank"),
        "vector_rank": it.get("vector_rank"),
        "lexical_score": it.get("lexical_score"),
        "vector_score": it.get("vector_score"),

        # rerank debugging
        "rerank_score": it.get("rerank_score"),
    }


def classify_intent(q: str) -> str:
    ql = (q or "").lower()

    if "investment objective" in ql or re.search(r"\bwhat('?s)?\s+the\s+fund('?s)?\s+objective\b", ql):
        return "objective"

    if "expense ratio" in ql or "fees" in ql or "expenses" in ql or re.search(r"\bcost(s)?\b", ql):
        return "fees"

    if "principal risks" in ql or re.search(r"\brisk(s)?\b", ql):
        return "risks"

    if ql.startswith("what is ") or ql.startswith("define "):
        return "definition"

    return "general"


def section_boost(intent: str, chunk_section: Optional[str], chunk_text: str) -> float:
    sec = (chunk_section or "").lower()
    t = (chunk_text or "").lower()

    if intent == "objective":
        # strongest signal: section match
        if "investment objective" in sec:
            return 0.25
        # fallback: canonical sentence patterns (still generic, not Vanguard-specific)
        if re.search(r"\bthe fund seeks to\b", t) or "seeks to track" in t or "designed to track" in t:
            return 0.15

    if intent == "fees":
        if "fees" in sec or "expenses" in sec:
            return 0.25
        if "expense ratio" in t:
            return 0.15

    if intent == "risks":
        if "risk" in sec or "principal risks" in sec or "risk factors" in sec:
            return 0.25

    if intent == "definition":
        if "glossary" in sec:
            return 0.20

    return 0.0


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/dev/reset")
def dev_reset():
    if os.environ.get("ENV") != "dev":
        raise HTTPException(status_code=404, detail="Not found")

    with SessionLocal() as db:
        db.execute(delete(Chunk))
        db.execute(delete(Document))
        db.commit()

    # IMPORTANT: also reset Qdrant so tests/dev runs are deterministic.
    vstore.reset_collection(vector_size=embedder.dim())

    return {"status": "reset ok"}


@app.post("/documents")
def create_document(req: IngestRequest):
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="title cannot be empty")
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="content cannot be empty")

    doc_id = str(uuid.uuid4())

    raw_chunks = chunk_text(req.content)
    chunks = _normalize_chunks(raw_chunks)
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks produced from content")

    with SessionLocal() as db:
        db.add(Document(id=doc_id, title=req.title, content=req.content))
        db.flush()  # ensures doc exists for FK

        chunk_rows = []  # (chunk_id, chunk_index, text, section)
        for i, ch in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            text = ch["text"]
            section = ch.get("section") or "General"

            if HAS_CHUNK_META:
                db.add(
                    Chunk(
                        id=chunk_id,
                        document_id=doc_id,
                        chunk_index=i,
                        content=text,
                        meta={"section": section},
                    )
                )
            else:
                db.add(
                    Chunk(
                        id=chunk_id,
                        document_id=doc_id,
                        chunk_index=i,
                        content=text,
                    )
                )

            chunk_rows.append((chunk_id, i, text, section))

        texts = [c[2] for c in chunk_rows]
        vectors = embedder.embed(texts)

        points = []
        for (chunk_id, idx, _text, section), vec in zip(chunk_rows, vectors):
            payload = {"document_id": doc_id, "chunk_index": idx}
            # optional but nice: allows vectorstore-side filtering/debug later
            payload["section"] = section

            points.append({"id": chunk_id, "vector": vec, "payload": payload})

        vstore.upsert_chunks(points)
        db.commit()

    return {"id": doc_id, "chunks_created": len(chunks)}


@app.post("/documents/upload_pdf")
def upload_pdf_document(
    title: str = Form(...),
    file: UploadFile = File(...),
    max_pages: int | None = Form(default=None),
):
    if not title.strip():
        raise HTTPException(status_code=400, detail="title cannot be empty")

    filename = (file.filename or "").lower()
    if (file.content_type not in ("application/pdf", "application/x-pdf")) and not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="file must be a PDF")

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        pages = extract_pages_from_pdf_bytes(data, max_pages=max_pages)
    except PdfExtractError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # If *every* page is empty -> scanned PDF / no extractable text
    if not any(p.strip() for p in pages):
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in PDF. If this is a scanned PDF, OCR is required.",
        )

    doc_id = str(uuid.uuid4())

    # For Document.content, store the joined text (handy for debugging / display)
    full_text = "\n\n".join([p for p in pages if p and p.strip()]).strip()

    # Option B chunking: includes {"text","section","page"}
    chunks = chunk_pages(pages)

    with SessionLocal() as db:
        db.add(Document(id=doc_id, title=title, content=full_text))
        db.flush()

        chunk_rows = []  # (chunk_id, chunk_index, text, section, page)
        for i, ch in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            text = ch["text"]
            section = str(ch.get("section") or "General")
            page = int(ch.get("page") or 0)

            db.add(
                Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    chunk_index=i,
                    content=text,
                    meta={"section": section, "page": page},
                )
            )
            chunk_rows.append((chunk_id, i, text, section, page))

        texts = [c[2] for c in chunk_rows]
        vectors = embedder.embed(texts)

        points = []
        for (chunk_id, idx, _text, section, page), vec in zip(chunk_rows, vectors):
            points.append(
                {
                    "id": chunk_id,
                    "vector": vec,
                    "payload": {
                        "document_id": doc_id,
                        "chunk_index": idx,
                        "section": section,
                        "page": page,
                    },
                }
            )
        vstore.upsert_chunks(points)

        db.commit()

    return {"id": doc_id, "chunks_created": len(chunks)}



@app.post("/search_vector")
def search_vector(req: VectorSearchRequest):
    qvec = embedder.embed([req.query])[0]
    hits = vstore.search(qvec, limit=req.top_k)

    hit_ids = [str(h.id) for h in hits]
    with SessionLocal() as db:
        if HAS_CHUNK_META:
            rows = db.execute(
                select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content, Chunk.meta).where(
                    Chunk.id.in_(hit_ids)
                )
            ).all()
            row_map = {
                r[0]: {
                    "chunk_id": r[0],
                    "document_id": r[1],
                    "chunk_index": r[2],
                    "content_preview": r[3][:200],
                    "meta": r[4] or {},
                }
                for r in rows
            }
        else:
            rows = db.execute(
                select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content).where(Chunk.id.in_(hit_ids))
            ).all()
            row_map = {
                r[0]: {
                    "chunk_id": r[0],
                    "document_id": r[1],
                    "chunk_index": r[2],
                    "content_preview": r[3][:200],
                    "meta": {},
                }
                for r in rows
            }

    results = []
    for h in hits:
        cid = str(h.id)
        if cid in row_map:
            results.append({"score": float(h.score), **row_map[cid]})

    return results


@app.get("/documents")
def list_documents():
    with SessionLocal() as db:
        rows = db.execute(select(Document.id, Document.title)).all()
    return [{"id": r[0], "title": r[1]} for r in rows]


@app.get("/documents/{doc_id}/chunks")
def list_chunks(doc_id: str):
    with SessionLocal() as db:
        if HAS_CHUNK_META:
            rows = db.execute(
                select(Chunk.id, Chunk.chunk_index, Chunk.content, Chunk.meta)
                .where(Chunk.document_id == doc_id)
                .order_by(Chunk.chunk_index)
            ).all()
            return [{"id": r[0], "chunk_index": r[1], "content": r[2], "meta": r[3] or {}} for r in rows]

        rows = db.execute(
            select(Chunk.id, Chunk.chunk_index, Chunk.content)
            .where(Chunk.document_id == doc_id)
            .order_by(Chunk.chunk_index)
        ).all()
        return [{"id": r[0], "chunk_index": r[1], "content": r[2], "meta": {}} for r in rows]


@app.post("/search")
def search(req: SearchRequest):
    with SessionLocal() as db:
        rows = db.execute(select(Chunk.id, Chunk.document_id, Chunk.chunk_index, Chunk.content)).all()

    ranked = rank_chunks(req.query, rows, top_k=req.top_k)
    return [
        {
            "score": s,
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "chunk_index": idx,
            "content_preview": content[:200],
        }
        for (s, chunk_id, doc_id, idx, content) in ranked
    ]


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    t0 = perf_counter()
    timings: Dict[str, float] = {}

    # --- Lexical
    t = perf_counter()
    with SessionLocal() as db:
        lex = lexical_search(db, req.query, top_k=req.lexical_k, doc_ids=req.doc_ids)
    timings["lexical_ms"] = (perf_counter() - t) * 1000.0

    # --- Vector
    t = perf_counter()
    qvec = embedder.embed([req.query])[0]
    vhits = vstore.search(qvec, limit=req.vector_k, doc_ids=req.doc_ids)
    vec = [{"chunk_id": str(h.id), "score": float(h.score)} for h in vhits]
    timings["vector_ms"] = (perf_counter() - t) * 1000.0

    # --- Fuse (RRF)
    t = perf_counter()
    fused = rrf_fuse(lex, vec, k=60)
    fused = fused[: req.candidates_k]
    timings["fuse_ms"] = (perf_counter() - t) * 1000.0

    # --- Fetch full chunk records
    t = perf_counter()
    ids = [x["chunk_id"] for x in fused]
    with SessionLocal() as db:
        chunk_map = fetch_chunks(db, ids)

    candidates: List[Dict[str, Any]] = []
    for x in fused:
        cid = x["chunk_id"]
        if cid in chunk_map:
            candidates.append({**x, **chunk_map[cid]})
    timings["fetch_ms"] = (perf_counter() - t) * 1000.0

    # --- Rerank (optional, capped)
    t = perf_counter()
    rerank_cap_used = None

    if req.use_rerank and reranker is not None:
        cap = min(req.candidates_k, MAX_RERANK_CANDIDATES)
        rerank_cap_used = cap
        candidates_for_rerank = candidates[:cap]

        candidates_for_rerank = reranker.rerank(req.query, candidates_for_rerank)
        for it in candidates_for_rerank:
            it["final_score"] = it["rerank_score"]

        candidates = candidates_for_rerank
    else:
        for it in candidates:
            it["final_score"] = it.get("rrf_score", 0.0)

    timings["rerank_ms"] = (perf_counter() - t) * 1000.0

    # --- Intent + section boost (generic, doc-structure-aware)
    intent = classify_intent(req.query)
    for it in candidates:
        sec = (it.get("meta") or {}).get("section")
        text = it.get("content") or ""
        it["final_score"] = float(it.get("final_score", 0.0)) + section_boost(intent, sec, text)

    # Baseline top-k (score-only) for redundancy comparison
    baseline = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[: req.top_k]

    # --- MMR (optional)
    t = perf_counter()
    dropped_duplicates = None
    baseline_redundant = None
    selected_redundant = None
    vectors_found = None
    vectors_missing = None
    selected_by = "score"

    if req.use_mmr:
        selected_by = "mmr"

        want = [it["chunk_id"] for it in candidates]
        vecs = vstore.get_vectors(want)

        found = sum(1 for cid in want if cid in vecs and vecs[cid] is not None)
        missing = len(want) - found
        vectors_found = found
        vectors_missing = missing

        selected = mmr_select(
            candidates=candidates,
            vectors=vecs,
            top_k=req.top_k,
            lam=req.mmr_lambda,
        )

        baseline_redundant = count_redundant_items(baseline, vecs, threshold=req.mmr_dup_threshold)
        selected_redundant = count_redundant_items(selected, vecs, threshold=req.mmr_dup_threshold)
        dropped_duplicates = max(0, baseline_redundant - selected_redundant)

        out = selected
    else:
        out = baseline

    timings["mmr_ms"] = (perf_counter() - t) * 1000.0
    timings["total_ms"] = (perf_counter() - t0) * 1000.0

    results = [format_result(it, selected_by=selected_by) for it in out]

    meta = {
        "use_mmr": req.use_mmr,
        "mmr_lambda": req.mmr_lambda if req.use_mmr else None,
        "mmr_dup_threshold": req.mmr_dup_threshold if req.use_mmr else None,
        "dropped_duplicates": dropped_duplicates if req.use_mmr else None,
        "use_rerank": bool(req.use_rerank and reranker is not None),
        "rerank_candidates_cap": rerank_cap_used,
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
        "baseline_redundant": baseline_redundant if req.use_mmr else None,
        "selected_redundant": selected_redundant if req.use_mmr else None,
        "vectors_found": vectors_found if req.use_mmr else None,
        "vectors_missing": vectors_missing if req.use_mmr else None,
    }

    if req.include_meta:
        return {"results": results, "meta": meta}

    return results


@app.post("/answer")
def answer(req: AnswerRequest):
    # Defaults tuned for "demo that feels good":
    # - rerank on by default (better quality)
    # - mmr off by default (less surprising results; can enable as a knob)
    use_rerank = req.use_rerank if req.use_rerank is not None else True
    use_mmr = req.use_mmr if req.use_mmr is not None else False

    retrieve_req = RetrieveRequest(
        query=req.query,
        top_k=req.top_k,
        lexical_k=req.lexical_k,
        vector_k=req.vector_k,
        candidates_k=req.candidates_k,
        use_mmr=use_mmr,
        mmr_lambda=req.mmr_lambda,
        mmr_dup_threshold=req.mmr_dup_threshold,
        use_rerank=use_rerank,
        include_meta=True,
        doc_ids=req.doc_ids,
    )

    payload = retrieve(retrieve_req)
    if isinstance(payload, dict):
        results = payload.get("results", [])
        meta = payload.get("meta", {})
    else:
        results = payload or []
        meta = {}


    if not results:
        resp = {"answer": "I don't know based on the provided sources.", "citations": []}
        if req.include_sources:
            resp["sources"] = []
        if req.include_meta:
            resp["retrieval_meta"] = meta
        return resp

    # Enrich results with FULL chunk content for grounding (prompt should not be based on previews)
    ids = [r["chunk_id"] for r in results if r.get("chunk_id")]
    with SessionLocal() as db:
        full_map = fetch_chunks(db, ids)

    for r in results:
        cid = r.get("chunk_id")
        if cid and cid in full_map:
            r["content"] = full_map[cid]["content"]
            r["meta"] = full_map[cid].get("meta") or r.get("meta") or {}

    # Build prompt with numbered sources (now prefers "content" over "content_preview")
    sources_text, idx_map = build_sources_block(results, max_chars_per_source=req.max_source_chars)

    system = (
        "You are a production RAG assistant.\n"
        "Rules:\n"
        "- Use ONLY the SOURCES provided.\n"
        f"- If the SOURCES do not contain enough info, reply with exactly: {REFUSAL_PREFIX}\n"
        "- Include citations like [1] for claims you make (at least one citation in the answer).\n"
        "- Do not cite numbers that do not exist in SOURCES.\n"
        "- Be concise.\n"
    )


    if req.strict_refusal:
        system += (
            "\nSTRICT MODE:\n"
            "- This mode ONLY applies when the user is asking for a DEFINITION of a term.\n"
            "- If the question asks to DEFINE a term, only answer if the SOURCES explicitly define it "
            "(e.g., 'X is defined as', 'X means', 'Definition: X').\n"
            f"- If the term is not explicitly defined in SOURCES, reply with exactly: {REFUSAL_PREFIX}\n"
        )


    user = f"{sources_text}\n\nUSER QUESTION:\n{req.query}\n\nAnswer with citations."

    try:
        llm = get_llm_provider()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    out = llm.generate(system=system, user=user).text

    # Enforce strict refusal contract (no extra text allowed)
    if _is_refusal(out):
        return {"answer": REFUSAL_PREFIX, "citations": []}


    # Validate grounding; if invalid, do ONE repair retry; else fail loudly with a code
    vr = validate_grounded_answer(out, max_source_index=len(results))
    if not vr.ok:
        err_summary = "; ".join(vr.errors[:3]) if vr.errors else "invalid citations"
        repair_user = (
            f"{sources_text}\n\n"
            f"USER QUESTION:\n{req.query}\n\n"
            f"Your previous answer had citation/grounding problems: {err_summary}\n\n"
            f"PREVIOUS ANSWER:\n{out}\n\n"
            "Rewrite the answer so that:\n"
            "- You ONLY use information from SOURCES.\n"
            "- Include citations like [1] for claims you make (at least one citation in the answer).\n"
            "- You ONLY cite valid source numbers (1..N).\n"
            "- Do NOT add new facts beyond what you already stated.\n"
            "- Keep it concise.\n"
        )

        out2 = llm.generate(system=system, user=repair_user).text

        if _is_refusal(out2):
            return {"answer": REFUSAL_PREFIX, "citations": []}

        vr2 = validate_grounded_answer(out2, max_source_index=len(results))
        if not vr2.ok:
            return {"answer": REFUSAL_PREFIX, "citations": []}
        out = out2

    # Map citations back to chunk metadata
    cited_nums = extract_citation_numbers(out)
    citations = []
    for n in cited_nums:
        if n in idx_map:
            r = idx_map[n]
            citations.append(
                {
                    "source_index": n,
                    "chunk_id": r.get("chunk_id"),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                }
            )

    resp = {"answer": out, "citations": citations}

    if req.include_sources:
        sources = []
        for i, r in enumerate(results, start=1):
            content = (r.get("content") or r.get("content_preview") or "").strip()
            content = content[: req.max_source_chars]
            sources.append(
                {
                    "source_index": i,
                    "chunk_id": r.get("chunk_id"),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                    "meta": r.get("meta") or {},
                    "content": content,
                }
            )
        resp["sources"] = sources

    if req.include_meta:
        resp["retrieval_meta"] = meta

    return resp
