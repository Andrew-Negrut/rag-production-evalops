from typing import List, Dict, Any
import math

def cosine(a, b) -> float:
    # vectors are already normalized, but keep this safe
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return dot / (na * nb)

def mmr_select(
    candidates: List[Dict[str, Any]],
    vectors: Dict[str, List[float]],
    top_k: int = 5,
    lam: float = 0.7,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    remaining = candidates[:]

    while remaining and len(selected) < top_k:
        best = None
        best_score = -1e9

        for item in remaining:
            cid = item["chunk_id"]
            rel = item.get("final_score", item.get("rrf_score", 0.0))
            v = vectors.get(cid)
            if v is None:
                score = rel
            else:
                max_sim = 0.0
                for s in selected:
                    sv = vectors.get(s["chunk_id"])
                    if sv is not None:
                        max_sim = max(max_sim, cosine(v, sv))
                score = lam * rel - (1 - lam) * max_sim

            if score > best_score:
                best_score = score
                best = item

        selected.append(best)
        remaining = [x for x in remaining if x["chunk_id"] != best["chunk_id"]]

    return selected
