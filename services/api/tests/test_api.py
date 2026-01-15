import os
import pytest
from fastapi.testclient import TestClient

# Set env BEFORE importing the app/db (db.py reads DATABASE_URL at import time)
# os.environ["DATABASE_URL"] = "sqlite+pysqlite:///./test.db"
os.environ["ENV"] = "dev"

from app.main import app
from app.db import Base, engine


@pytest.fixture(scope="session", autouse=True)
def create_test_tables():
    """
    Create tables once for the SQLite test database.
    (In a more advanced setup, we'd run Alembic migrations here.)
    """
    Base.metadata.create_all(bind=engine)


@pytest.fixture()
def client():
    """
    Fresh TestClient per test so startup/shutdown runs cleanly.
    Also reset the DB via the API to ensure isolation between tests.
    """
    with TestClient(app) as c:
        # ensure a clean state for each test
        r = c.post("/dev/reset")
        assert r.status_code == 200
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_document_creates_chunks_and_search_works(client):
    payload = {"title": "Chunk test", "content": "This is a test. " * 300}
    r = client.post("/documents", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["chunks_created"] > 0
    doc_id = data["id"]

    r = client.get(f"/documents/{doc_id}/chunks")
    assert r.status_code == 200
    chunks = r.json()
    assert len(chunks) == data["chunks_created"]

    r = client.post("/search", json={"query": "test", "top_k": 3})
    assert r.status_code == 200
    results = r.json()
    assert len(results) > 0
    assert results[0]["score"] > 0


def test_validation_empty_content(client):
    r = client.post("/documents", json={"title": "x", "content": "   "})
    assert r.status_code == 400
