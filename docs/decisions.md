# Decisions

This document captures the minimal stack choices to reduce churn.

## API
- FastAPI + Pydantic: fast iteration, typed request/response models, good ecosystem.

## Background jobs
- Not implemented yet. Planned: Celery + Redis for ingestion/rebuild jobs when needed.

## Storage
- Postgres: app state, documents, chunks, and lexical search via full-text index.

## Vector search
- Qdrant (Docker): simple local setup, production-capable API, good hybrid pairing with Postgres.

## BM25 / Lexical
- Postgres full-text search: avoids extra services while providing BM25-like lexical retrieval.

## Reranker
- sentence-transformers CrossEncoder (optional): quality boost with a capped candidate set.

## Observability
- OpenTelemetry → Jaeger (tracing)
- Prometheus → Grafana (metrics + dashboard)

## CI
- GitHub Actions with ruff, mypy, pytest, pip-audit, bandit (to be wired).
