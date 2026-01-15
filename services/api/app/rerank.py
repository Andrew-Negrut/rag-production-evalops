from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = [(query, it["content"]) for it in items]
        scores = self.model.predict(pairs)
        for it, s in zip(items, scores):
            it["rerank_score"] = float(s)
        items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return items
