import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml

@dataclass(frozen=True)
class Settings:
    env: str
    database_url: str
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str
    rag_config: Dict[str, Any]
    chunk_size: int
    chunk_overlap: int
    retrieve_defaults: Dict[str, Any]
    answer_defaults: Dict[str, Any]
    rrf_k: int
    rerank_model: str


def _load_rag_config() -> Dict[str, Any]:
    path = os.environ.get("RAG_CONFIG_PATH", "config/rag.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def get_settings() -> Settings:
    rag_config = _load_rag_config()
    chunking = rag_config.get("chunking") or {}
    retrieval = rag_config.get("retrieval_defaults") or {}
    answer = rag_config.get("answer_defaults") or {}
    models = rag_config.get("models") or {}

    chunk_size = int(chunking.get("chunk_size", 800))
    chunk_overlap = int(chunking.get("overlap", 100))

    retrieve_defaults = {
        "top_k": int(retrieval.get("top_k", 5)),
        "lexical_k": int(retrieval.get("lexical_k", 20)),
        "vector_k": int(retrieval.get("vector_k", 20)),
        "candidates_k": int(retrieval.get("candidates_k", 50)),
        "use_mmr": bool(retrieval.get("use_mmr", True)),
        "use_rerank": bool(retrieval.get("use_rerank", False)),
    }

    answer_defaults = {
        "top_k": int(answer.get("top_k", 5)),
        "lexical_k": int(answer.get("lexical_k", 20)),
        "vector_k": int(answer.get("vector_k", 20)),
        "candidates_k": int(answer.get("candidates_k", 50)),
        "use_mmr": bool(answer.get("use_mmr", False)),
        "use_rerank": bool(answer.get("use_rerank", True)),
    }

    rrf_k = int(retrieval.get("rrf_k", 60))
    rerank_model = str(models.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    return Settings(
        env=os.environ.get("ENV", "dev"),
        database_url=os.environ["DATABASE_URL"],
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "chunks"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        rag_config=rag_config,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieve_defaults=retrieve_defaults,
        answer_defaults=answer_defaults,
        rrf_k=rrf_k,
        rerank_model=rerank_model,
    )
