from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import httpx


def _strip_html(text: str) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _iter_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return

    for p in sorted(source.rglob("*")):
        if p.is_file():
            yield p


def _record_to_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in record.items():
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}: {v}")
    return "\n".join(parts).strip()


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _load_csv_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _post_document(client: httpx.Client, base_url: str, title: str, content: str) -> None:
    resp = client.post(
        f"{base_url}/documents",
        json={"title": title, "content": content},
    )
    resp.raise_for_status()


def _post_pdf(client: httpx.Client, base_url: str, title: str, path: Path) -> None:
    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/pdf")}
        data = {"title": title}
        resp = client.post(f"{base_url}/documents/upload_pdf", data=data, files=files)
        resp.raise_for_status()


def _ingest_file(client: httpx.Client, base_url: str, path: Path) -> Tuple[int, int]:
    ext = path.suffix.lower()
    ingested = 0
    skipped = 0

    if ext == ".pdf":
        _post_pdf(client, base_url, title=path.stem, path=path)
        return (1, 0)

    if ext in {".txt", ".md"}:
        content = path.read_text(encoding="utf-8")
        _post_document(client, base_url, title=path.stem, content=content)
        return (1, 0)

    if ext in {".html", ".htm"}:
        content = _strip_html(path.read_text(encoding="utf-8"))
        if content:
            _post_document(client, base_url, title=path.stem, content=content)
            return (1, 0)
        return (0, 1)

    if ext in {".json", ".jsonl"}:
        records = _load_json_records(path)
        for idx, record in enumerate(records, start=1):
            content = _record_to_text(record)
            if not content:
                skipped += 1
                continue
            title = f"{path.stem}-{idx}"
            _post_document(client, base_url, title=title, content=content)
            ingested += 1
        return (ingested, skipped)

    if ext == ".csv":
        records = _load_csv_records(path)
        for idx, record in enumerate(records, start=1):
            content = _record_to_text(record)
            if not content:
                skipped += 1
                continue
            title = f"{path.stem}-{idx}"
            _post_document(client, base_url, title=title, content=content)
            ingested += 1
        return (ingested, skipped)

    return (0, 1)


def run_ingest(source: Path, base_url: str, reset: bool) -> Tuple[int, int]:
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    base_url = base_url.rstrip("/")
    ingested = 0
    skipped = 0

    with httpx.Client(timeout=60) as client:
        if reset:
            resp = client.post(f"{base_url}/dev/reset")
            resp.raise_for_status()

        for path in _iter_files(source):
            i, s = _ingest_file(client, base_url, path)
            ingested += i
            skipped += s

    return (ingested, skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG API.")
    parser.add_argument("--source", required=True, help="File or directory to ingest (e.g., ./data/raw)")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--reset", action="store_true", help="Call /dev/reset before ingesting.")
    args = parser.parse_args()

    ingested, skipped = run_ingest(
        source=Path(args.source),
        base_url=args.base_url,
        reset=args.reset,
    )
    print(f"[ingest] ingested={ingested} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
