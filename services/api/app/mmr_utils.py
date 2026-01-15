from __future__ import annotations

from typing import Dict, List, Sequence
import math


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def count_redundant_items(
    items: List[dict],
    vectors_by_chunk_id: Dict[str, Sequence[float]],
    threshold: float = 0.95,
) -> int:
    kept_vecs: List[Sequence[float]] = []
    redundant = 0

    for it in items:
        cid = it.get("chunk_id")
        if not cid:
            continue
        v = vectors_by_chunk_id.get(cid)
        if v is None:
            continue

        is_dup = any(cosine_sim(v, kv) > threshold for kv in kept_vecs)
        if is_dup:
            redundant += 1
        else:
            kept_vecs.append(v)

    return redundant
