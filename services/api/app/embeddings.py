# from typing import List
# from sentence_transformers import SentenceTransformer

# class EmbeddingProvider:
#     def __init__(self, model_name: str):
#         self.model = SentenceTransformer(model_name)

#     def embed(self, texts: List[str]) -> List[List[float]]:
#         vectors = self.model.encode(
#             texts,
#             normalize_embeddings=True,
#             batch_size=32,
#             show_progress_bar=False,
#         )
#         return vectors.tolist()

#     def dim(self) -> int:
#         return self.model.get_sentence_embedding_dimension()

from __future__ import annotations

import hashlib
from typing import List

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore


class EmbeddingProvider:
    """
    Two modes:

    1) Normal: SentenceTransformer(model_name) when model_name != "hash"
    2) CI/Test: deterministic hash embeddings when model_name == "hash"
       - no downloads, no network, stable vectors
    """

    def __init__(self, model_name: str):
        self.model_name = (model_name or "").strip()

        if self.model_name.lower() == "hash":
            self._dim = 128  # small but sufficient for smoke tests
            self.model = None
            return

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not available, but EMBEDDING_MODEL != 'hash'. "
                "Either install sentence-transformers or set EMBEDDING_MODEL=hash."
            )

        self.model = SentenceTransformer(self.model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model_name.lower() == "hash":
            return [self._hash_embed(t) for t in texts]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def dim(self) -> int:
        return self._dim

    def _hash_embed(self, text: str) -> List[float]:
        """
        Deterministic embedding:
        - map tokens to buckets via sha256
        - build a signed bag-of-words style vector
        - L2 normalize
        """
        v = [0.0] * self._dim
        toks = (text or "").lower().split()
        for tok in toks:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            v[idx] += sign

        norm = sum(x * x for x in v) ** 0.5
        if norm > 0:
            v = [x / norm for x in v]
        return v
