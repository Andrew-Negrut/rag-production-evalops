from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore

class Reranker:
    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not available, but reranking was enabled. "
                "Install the ml extra or set ENABLE_RERANK=false."
            )
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = [(query, it["content"]) for it in items]
        scores = self.model.predict(pairs)
        for it, s in zip(items, scores):
            it["rerank_score"] = float(s)
        items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return items
