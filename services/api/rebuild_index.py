from __future__ import annotations

import argparse
import os
from pathlib import Path

from services.api.ingest import run_ingest


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset and rebuild the index deterministically.")
    parser.add_argument("--source", default="data/seed", help="Seed dataset path (default: data/seed)")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    ingested, skipped = run_ingest(
        source=Path(args.source),
        base_url=args.base_url,
        reset=True,
    )
    print(f"[rebuild] ingested={ingested} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
