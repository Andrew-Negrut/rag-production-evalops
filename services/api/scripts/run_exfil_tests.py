from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exfiltration tests against /answer.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for results.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"),
    )
    args = parser.parse_args()

    dataset = "services/api/safety/exfiltration.jsonl"
    base_url = args.base_url
    cmd = [
        sys.executable,
        "services/api/scripts/run_injection_tests.py",
        "--dataset",
        dataset,
        "--base-url",
        base_url,
    ]
    if args.out_dir:
        cmd.extend(["--out-dir", args.out_dir])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
