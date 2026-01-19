# import os
# from fastapi.testclient import TestClient

# os.environ["ENV"] = "dev"
# os.environ["LLM_PROVIDER"] = "fake"

# from app.main import app
# from app.answer_validation import validate_grounded_answer


# def test_answer_returns_citations_and_sources_when_requested():
#     with TestClient(app) as c:
#         # deterministic reset (db + qdrant)
#         r = c.post("/dev/reset")
#         assert r.status_code == 200

#         # ingest a simple doc
#         payload = {"title": "Doc", "content": "This is a test document. " * 50}
#         r = c.post("/documents", json=payload)
#         assert r.status_code == 200

#         # call /answer
#         r = c.post("/answer", json={"query": "What is the document about?", "top_k": 3, "include_sources": True})
#         assert r.status_code == 200
#         data = r.json()

#         assert "answer" in data
#         assert "citations" in data
#         assert "sources" in data

#         # fake provider always uses [1]
#         assert "[1]" in data["answer"]
#         assert len(data["citations"]) >= 1
#         assert data["citations"][0]["source_index"] == 1

#         # sources should contain content
#         assert len(data["sources"]) >= 1
#         assert "content" in data["sources"][0]
#         assert "test document" in data["sources"][0]["content"].lower()


# def test_answer_empty_index_refuses_cleanly():
#     with TestClient(app) as c:
#         r = c.post("/dev/reset")
#         assert r.status_code == 200

#         r = c.post("/answer", json={"query": "Anything here?", "top_k": 3})
#         assert r.status_code == 200
#         data = r.json()

#         assert data["citations"] == []
#         assert "don't know" in data["answer"].lower()


# def test_answer_validator_basics():
#     ok = validate_grounded_answer("Hello [1]\n\nWorld [1, 2]", max_source_index=2)
#     assert ok.ok

#     # With the current validator, per-paragraph citations are NOT required.
#     ok2 = validate_grounded_answer("Hello\n\nWorld [1]", max_source_index=2)
#     assert ok2.ok

#     bad2 = validate_grounded_answer("Hello\n\nWorld", max_source_index=2)
#     assert not bad2.ok

#     # assert any("Paragraph 1" in e for e in bad.errors)

#     refusal = validate_grounded_answer("I don't know based on the provided sources.", max_source_index=3)
#     assert refusal.ok


import os
from fastapi.testclient import TestClient

# Ensure env vars are set BEFORE importing app.main (so providers initialize correctly)
os.environ["ENV"] = "dev"
os.environ["LLM_PROVIDER"] = "fake"

# Optional but strongly recommended for deterministic/local+CI runs (no model downloads)
os.environ.setdefault("EMBEDDING_MODEL", "hash")
os.environ.setdefault("ENABLE_RERANK", "false")

from app.main import app
from app.answer_validation import validate_grounded_answer


def test_answer_returns_citations_and_sources_when_requested():
    with TestClient(app) as c:
        # deterministic reset (db + qdrant)
        r = c.post("/dev/reset")
        assert r.status_code == 200

        # ingest a simple doc
        payload = {"title": "Doc", "content": "This is a test document. " * 50}
        r = c.post("/documents", json=payload)
        assert r.status_code == 200

        # call /answer
        r = c.post(
            "/answer",
            json={"query": "What is the document about?", "top_k": 3, "include_sources": True},
        )
        assert r.status_code == 200
        data = r.json()

        assert "answer" in data
        assert "citations" in data
        assert "sources" in data

        # fake provider should produce at least one citation
        assert "[1]" in data["answer"]
        assert len(data["citations"]) >= 1
        assert data["citations"][0]["source_index"] == 1

        # sources should contain content
        assert len(data["sources"]) >= 1
        assert "content" in data["sources"][0]
        assert "test document" in data["sources"][0]["content"].lower()


def test_answer_empty_index_refuses_cleanly():
    with TestClient(app) as c:
        r = c.post("/dev/reset")
        assert r.status_code == 200

        r = c.post("/answer", json={"query": "Anything here?", "top_k": 3})
        assert r.status_code == 200
        data = r.json()

        assert data["citations"] == []
        assert "don't know" in data["answer"].lower()


def test_answer_validator_basics():
    ok = validate_grounded_answer("Hello [1]\n\nWorld [1, 2]", max_source_index=2)
    assert ok.ok

    # Per-paragraph citations are NOT required by the current validator (only overall citations).
    ok2 = validate_grounded_answer("Hello\n\nWorld [1]", max_source_index=2)
    assert ok2.ok

    bad2 = validate_grounded_answer("Hello\n\nWorld", max_source_index=2)
    assert not bad2.ok

    # Option B: refusals are considered valid without citations.
    refusal = validate_grounded_answer("I don't know based on the provided sources.", max_source_index=3)
    assert refusal.ok
