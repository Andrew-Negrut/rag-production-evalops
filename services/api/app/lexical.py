from typing import List, Dict, Any, Optional
import re
from sqlalchemy import select, func, text, bindparam
from app.models import Chunk

# a small stoplist to remove fluff words that often break strict lexical matching
STOP = {
    "what", "which", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "for", "to", "of", "and", "or", "in", "on", "with",
    "system",  # optional; helps your exact example
}

def _keywords(query: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9]+", query.lower())
    toks = [t for t in toks if len(t) >= 3 and t not in STOP]
    # dedupe but keep order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def lexical_search(db, query: str, top_k: int = 10, doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Primary: strict websearch_to_tsquery (fast + good for keyword-style queries)
    Fallback: OR query across extracted keywords (more forgiving for natural language)
    """
    # ---- strict pass ----
    tsq = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Chunk.search_tsv, tsq)

    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.content,
            rank.label("score"),
        )
        .where(Chunk.search_tsv.op("@@")(tsq))
    )

    if doc_ids:
        stmt = stmt.where(Chunk.document_id.in_(doc_ids))

    rows = db.execute(
        stmt.order_by(rank.desc()).limit(top_k)
    ).all()

    if len(rows) >= min(3, top_k):  # enough results, use strict
        return [
            {
                "chunk_id": r[0],
                "document_id": r[1],
                "chunk_index": r[2],
                "content": r[3],
                "lexical_score": float(r[4]),
            }
            for r in rows
        ]

    # ---- loose fallback ----
    kws = _keywords(query)
    if not kws:
        return []

    qstr = " | ".join(kws)  # OR across tokens

    tsq_loose = func.to_tsquery("english", bindparam("q"))
    rank_loose = func.ts_rank_cd(Chunk.search_tsv, tsq_loose)

    stmt2 = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.content,
            rank_loose.label("score"),
        )
        .where(Chunk.search_tsv.op("@@")(tsq_loose))
    )

    if doc_ids:
        stmt2 = stmt2.where(Chunk.document_id.in_(doc_ids))

    rows2 = db.execute(
        stmt2.order_by(rank_loose.desc()).limit(top_k),
        {"q": qstr},
    ).all()

    return [
        {
            "chunk_id": r[0],
            "document_id": r[1],
            "chunk_index": r[2],
            "content": r[3],
            "lexical_score": float(r[4]),
        }
        for r in rows2
    ]
