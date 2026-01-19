from typing import Dict, List, Any

def rrf_fuse(
    lexical: List[Dict[str, Any]],
    vector: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}

    # Add lexical ranks
    for rank, item in enumerate(lexical, start=1):
        cid = item["chunk_id"]
        fused.setdefault(cid, {"chunk_id": cid})
        fused[cid]["lexical_rank"] = rank
        fused[cid]["lexical_score"] = item.get("lexical_score")

    # Add vector ranks
    for rank, item in enumerate(vector, start=1):
        cid = item["chunk_id"]
        fused.setdefault(cid, {"chunk_id": cid})
        fused[cid]["vector_rank"] = rank
        fused[cid]["vector_score"] = item.get("score")

    # Compute fused score
    results = []
    for cid, d in fused.items():
        score = 0.0
        if "lexical_rank" in d:
            score += 1.0 / (k + d["lexical_rank"])
        if "vector_rank" in d:
            score += 1.0 / (k + d["vector_rank"])
        d["rrf_score"] = score
        results.append(d)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results
