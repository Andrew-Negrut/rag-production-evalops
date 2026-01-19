from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class VectorStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == self.collection for c in collections):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )

    def reset_collection(self, vector_size: int) -> None:
        """
        Deterministic reset for dev/tests: delete and recreate collection.
        """
        try:
            self.client.delete_collection(collection_name=self.collection)
        except Exception:
            # If it doesn't exist or Qdrant is starting up, we keep going.
            pass

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=vector_size,
                distance=qm.Distance.COSINE,
            ),
        )

    def upsert_chunks(self, points: List[Dict[str, Any]]) -> None:
        """
        points: [{ "id": <chunk_id>, "vector": [...], "payload": {...}}]
        """
        qpoints = [
            qm.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        self.client.upsert(collection_name=self.collection, points=qpoints)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        doc_ids: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[qm.ScoredPoint]:
        must = []
        if doc_ids:
            must.append(
                qm.FieldCondition(
                    key="document_id",
                    match=qm.MatchAny(any=doc_ids),
                )
            )
        if tenant_id:
            must.append(
                qm.FieldCondition(
                    key="tenant_id",
                    match=qm.MatchValue(value=tenant_id),
                )
            )
        qfilter = qm.Filter(must=must) if must else None

        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
            query_filter=qfilter,
        )


    def get_vectors(self, ids: List[str]) -> Dict[str, List[float]]:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=ids,
            with_vectors=True,
            with_payload=False,
        )
        out: Dict[str, List[float]] = {}
        for p in points:
            out[str(p.id)] = p.vector  # type: ignore[attr-defined]
        return out
