from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingProvider:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
