import argparse
import json
import os
import time

import httpx

from app.answer_validation import validate_grounded_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--sleep-ms", type=int, default=0)
    args = p.parse_args()

    base = args.base_url.rstrip("/")

    client = httpx.Client(timeout=30.0)

    # Reset and seed minimal data to avoid "empty index" runs.
    r = client.post(f"{base}/dev/reset")
    if r.status_code != 200:
        raise SystemExit(f"/dev/reset failed: {r.status_code} {r.text}")

    seed_doc = {"title": "Seed", "content": "This is a test document about stability. " * 80}
    r = client.post(f"{base}/documents", json=seed_doc)
    if r.status_code != 200:
        raise SystemExit(f"/documents failed: {r.status_code} {r.text}")

    bad = 0
    errors = {}

    for i in range(args.n):
        rr = client.post(
            f"{base}/answer",
            json={
                "query": "What is this document about?",
                "top_k": 5,
                "include_sources": True,
            },
        )

        if rr.status_code != 200:
            bad += 1
            key = f"http_{rr.status_code}"
            errors[key] = errors.get(key, 0) + 1
        else:
            data = rr.json()
            sources = data.get("sources", [])
            answer = data.get("answer", "")
            vr = validate_grounded_answer(answer, max_source_index=len(sources))

            if not vr.ok:
                bad += 1
                key = "ungrounded"
                errors[key] = errors.get(key, 0) + 1

        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    summary = {"n": args.n, "bad": bad, "errors": errors}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
