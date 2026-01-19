# Production RAG Service

## Overview
This repo implements a production‑oriented RAG service with ingestion, hybrid retrieval, grounded answers, eval harness, and observability.

## Performance & Cost
Numbers below are from a local dev run with `LLM_PROVIDER=fake`, `EMBEDDING_MODEL=hash`, and no cache.

- p95 latency (POST `/answer`): <= 0.05s (Prometheus histogram bucket)
- tokens in/out: 0 (fake provider does not emit token usage)
- cost estimate: $0.00 (cost per 1K tokens defaults to 0)
- cache hit rate: 0% (cache not implemented; hits are always 0)
- refusal rate: 0% in sample run (see `rag_answer_refusal_total / rag_answer_total`)

To view live metrics, open `http://localhost:8000/metrics` or the Grafana dashboard at `http://localhost:3000`.
