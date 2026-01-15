import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    env: str
    database_url: str
    qdrant_url: str
    qdrant_collection: str
    embedding_model: str

def get_settings() -> Settings:
    return Settings(
        env=os.environ.get("ENV", "dev"),
        database_url=os.environ["DATABASE_URL"],
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.environ.get("QDRANT_COLLECTION", "chunks"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
