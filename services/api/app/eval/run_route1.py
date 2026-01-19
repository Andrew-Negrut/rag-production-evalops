from __future__ import annotations

import json
import os
import sys
import time
import uuid
import hashlib
import subprocess
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import httpx
from datetime import datetime, timezone



DEFAULT_BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")

# Tune these once and keep them stable for regression tracking
RETRIEVE_PAYLOAD_DEFAULTS = {
    "top_k": 10,
    "lexical_k": 60,
    "vector_k": 60,
    "candidates_k": 120,
    "include_meta": True,
    "use_rerank": True,
    "use_mmr": False,
}

ANSWER_PAYLOAD_DEFAULTS = {
    "top_k": 10,
    "lexical_k": 60,
    "vector_k": 60,
    "candidates_k": 120,
    "use_rerank": True,
    "use_mmr": False,
    "include_sources": False,
    "include_meta": False,
}

REFUSAL_PREFIX = "I don't know based on the provided sources."

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_sha() -> str:
    # Prefer CI env var (works inside Docker/CI even if .git isn't present)
    sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
    if sha:
        return sha.strip()

    # Fallback: try git
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _is_git_dirty() -> bool:
    # Optional override for CI/container builds
    dirty_env = os.environ.get("GIT_DIRTY")
    if dirty_env is not None:
        return dirty_env.strip().lower() in ("1", "true", "yes")

    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
        return bool(out.decode("utf-8").strip())
    except Exception:
        # If we cannot determine, default to False (but SHA may be "unknown")
        return False


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.request(method.upper(), url, json=payload)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} calling {url}: {e.response.text}") from e
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to call {url}: {e}") from e


_DOC_KEY_RE = re.compile(r"^In\s+([A-Za-z0-9_]+)\b")


def _fetch_documents(base_url: str) -> List[Dict[str, Any]]:
    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(f"{base_url}/documents")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _extract_doc_key(query: str) -> Optional[str]:
    m = _DOC_KEY_RE.match((query or "").strip())
    return m.group(1) if m else None


@dataclass
class GoldenExample:
    id: str
    category: str
    query: str
    target_document_ids: List[str]
    answerable: bool


def _load_jsonl(path: str) -> List[GoldenExample]:
    out: List[GoldenExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                GoldenExample(
                    id=obj["id"],
                    category=obj.get("category", "uncategorized"),
                    query=obj["query"],
                    target_document_ids=list(obj["target_document_ids"]),
                    answerable=bool(obj["answerable"]),
                )
            )
    if not out:
        raise RuntimeError(f"No examples loaded from {path}")
    return out


def _any_result_from_targets(results: List[Dict[str, Any]], targets: List[str]) -> bool:
    tset = set(targets)
    for r in results:
        if r.get("document_id") in tset:
            return True
    return False


def _citation_doc_precision(citations: List[Dict[str, Any]], targets: List[str]) -> Tuple[float, int, int]:
    """
    Returns (precision, correct, total)
    Precision is computed over citations returned by /answer (doc_id match).
    """
    if not citations:
        return (1.0, 0, 0)
    tset = set(targets)
    total = 0
    correct = 0
    for c in citations:
        total += 1
        if c.get("document_id") in tset:
            correct += 1
    return (correct / total if total else 1.0, correct, total)


def _load_failure_taxonomy() -> Dict[str, str]:
    here = os.path.dirname(__file__)
    path = os.path.abspath(os.path.join(here, "..", "..", "eval", "failure_taxonomy.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _classify_failure(
    ex: GoldenExample,
    hit: bool,
    is_refusal: bool,
    citations_total: int,
    citations_correct: int,
) -> str:
    if not hit:
        return "retrieval_miss"
    if ex.answerable:
        if is_refusal:
            return "answerable_refusal"
        if citations_total < 1:
            return "answerable_no_citations"
        if citations_total > 0 and citations_correct == 0:
            return "citation_doc_mismatch"
    else:
        if not is_refusal:
            return "unanswerable_non_refusal"
        if citations_total > 0:
            return "unanswerable_has_citations"
    return "unknown"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.eval.run_route1 eval/golden.route1.jsonl [base_url]")
        return 2

    dataset_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_BASE_URL

    examples = _load_jsonl(dataset_path)
    docs = _fetch_documents(base_url)
    doc_id_by_title = {
        d.get("title"): d.get("id")
        for d in docs
        if isinstance(d, dict) and d.get("title") and d.get("id")
    }
    if doc_id_by_title:
        current_ids = set(doc_id_by_title.values())
        for ex in examples:
            if any(tid in current_ids for tid in ex.target_document_ids):
                continue
            key = _extract_doc_key(ex.query)
            if key and key in doc_id_by_title:
                ex.target_document_ids = [doc_id_by_title[key]]

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    failures: List[Dict[str, Any]] = []
    failure_taxonomy = _load_failure_taxonomy()
    failure_counts: Dict[str, int] = {}

    # Aggregates
    n = 0
    retrieval_hit = 0

    answer_correct_behavior = 0
    citation_precision_sum = 0.0
    citation_precision_count = 0

    refusal_expected = 0
    refusal_correct = 0

    by_cat: Dict[str, Dict[str, float]] = {}

    for ex in examples:
        n += 1

        # ---- /retrieve
        retrieve_payload = dict(RETRIEVE_PAYLOAD_DEFAULTS)
        retrieve_payload["query"] = ex.query
        retrieve_payload["doc_ids"] = ex.target_document_ids
        rj = _http_json("POST", f"{base_url}/retrieve", retrieve_payload)
        results = rj.get("results", rj)  # supports include_meta true/false
        if not isinstance(results, list):
            results = []

        hit = _any_result_from_targets(results[: int(retrieve_payload["top_k"])], ex.target_document_ids)
        retrieval_hit += 1 if hit else 0

        # ---- /answer
        answer_payload = dict(ANSWER_PAYLOAD_DEFAULTS)
        answer_payload["query"] = ex.query
        answer_payload["doc_ids"] = ex.target_document_ids
        answer_payload["strict_refusal"] = ("define" in ex.query.lower())
        aj = _http_json("POST", f"{base_url}/answer", answer_payload)

        answer_text = (aj.get("answer") or "").strip()
        citations = aj.get("citations") or []
        if not isinstance(citations, list):
            citations = []

        is_refusal = answer_text.startswith(REFUSAL_PREFIX)

        # Behavior rules:
        # - If answerable==False: should refuse AND citations should be empty.
        # - If answerable==True: should NOT refuse AND should have >=1 citation.
        ok = True
        if ex.answerable:
            if is_refusal:
                ok = False
            if len(citations) < 1:
                ok = False
        else:
            refusal_expected += 1
            if not is_refusal:
                ok = False
            if len(citations) != 0:
                ok = False
            if is_refusal:
                refusal_correct += 1

        # Citation precision (only meaningful when citations exist)
        prec, correct_c, total_c = _citation_doc_precision(citations, ex.target_document_ids)
        if total_c > 0:
            citation_precision_sum += prec
            citation_precision_count += 1

        if ok:
            answer_correct_behavior += 1
        else:
            failure_type = _classify_failure(
                ex,
                hit=hit,
                is_refusal=is_refusal,
                citations_total=total_c,
                citations_correct=correct_c,
            )
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
            failures.append(
                {
                    "id": ex.id,
                    "category": ex.category,
                    "query": ex.query,
                    "targets": ex.target_document_ids,
                    "answerable": ex.answerable,
                    "retrieve_hit@k": hit,
                    "answer_is_refusal": is_refusal,
                    "citations_total": total_c,
                    "citations_correct_doc": correct_c,
                    "citation_precision": prec,
                    "failure_type": failure_type,
                    "answer_preview": answer_text[:240],
                }
            )

        # category breakdown
        cat = ex.category or "uncategorized"
        if cat not in by_cat:
            by_cat[cat] = {"n": 0, "retrieval_hit": 0, "answer_ok": 0}
        by_cat[cat]["n"] += 1
        by_cat[cat]["retrieval_hit"] += 1 if hit else 0
        by_cat[cat]["answer_ok"] += 1 if ok else 0

    metrics = {
        "run_id": run_id,
        "base_url": base_url,
        "dataset": dataset_path,
        "n_examples": n,
        "retrieval_target_doc_recall@k": (retrieval_hit / n if n else 0.0),
        "answer_behavior_accuracy": (answer_correct_behavior / n if n else 0.0),
        "avg_citation_doc_precision": (citation_precision_sum / citation_precision_count if citation_precision_count else None),
        "refusal_accuracy_on_unanswerable": (refusal_correct / refusal_expected if refusal_expected else None),
        "by_category": {
            k: {
                "n": int(v["n"]),
                "retrieval_target_doc_recall@k": (v["retrieval_hit"] / v["n"] if v["n"] else 0.0),
                "answer_behavior_accuracy": (v["answer_ok"] / v["n"] if v["n"] else 0.0),
            }
            for k, v in by_cat.items()
        },
        "defaults": {
            "retrieve": RETRIEVE_PAYLOAD_DEFAULTS,
            "answer": ANSWER_PAYLOAD_DEFAULTS,
            "refusal_prefix": REFUSAL_PREFIX,
        },
    }

    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "dirty_git_tree": _is_git_dirty(),
        "dataset_path": dataset_path,
        "dataset_sha256": _sha256_file(dataset_path),
        "base_url": base_url,
        "retrieve_payload_defaults": RETRIEVE_PAYLOAD_DEFAULTS,
        "answer_payload_defaults": ANSWER_PAYLOAD_DEFAULTS,
        "key_metrics": {
            "n_examples": metrics["n_examples"],
            "retrieval_target_doc_recall@k": metrics["retrieval_target_doc_recall@k"],
            "answer_behavior_accuracy": metrics["answer_behavior_accuracy"],
            "avg_citation_doc_precision": metrics["avg_citation_doc_precision"],
            "refusal_accuracy_on_unanswerable": metrics["refusal_accuracy_on_unanswerable"],
        },
    }

    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # simple markdown report
    report_lines = []
    report_lines.append("# Route 1 Eval Report\n")
    report_lines.append(f"- run_id: `{run_id}`")
    report_lines.append(f"- base_url: `{base_url}`")
    report_lines.append(f"- dataset: `{dataset_path}`")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- examples: **{n}**")
    report_lines.append(f"- retrieval target-doc recall@k: **{metrics['retrieval_target_doc_recall@k']:.3f}**")
    report_lines.append(f"- answer behavior accuracy: **{metrics['answer_behavior_accuracy']:.3f}**")
    if metrics["avg_citation_doc_precision"] is not None:
        report_lines.append(f"- avg citation doc precision: **{metrics['avg_citation_doc_precision']:.3f}**")
    if metrics["refusal_accuracy_on_unanswerable"] is not None:
        report_lines.append(f"- refusal accuracy (unanswerable): **{metrics['refusal_accuracy_on_unanswerable']:.3f}**")
    report_lines.append("")
    report_lines.append("## By category")
    for cat, m in metrics["by_category"].items():
        report_lines.append(f"- **{cat}**: n={m['n']}, recall@k={m['retrieval_target_doc_recall@k']:.3f}, answer_ok={m['answer_behavior_accuracy']:.3f}")
    report_lines.append("")
    report_lines.append("## Failures (first 25)")
    for fitem in failures[:25]:
        report_lines.append(f"- `{fitem['id']}` [{fitem['category']}] hit@k={fitem['retrieve_hit@k']} refusal={fitem['answer_is_refusal']} cit_prec={fitem['citation_precision']:.3f} :: {fitem['answer_preview']}")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # also dump failures json for later “failure gallery”
    with open(os.path.join(run_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    failure_gallery = {
        "run_id": run_id,
        "dataset": dataset_path,
        "taxonomy": failure_taxonomy,
        "counts": failure_counts,
        "failures": failures,
    }
    with open(os.path.join(run_dir, "failure_gallery.json"), "w", encoding="utf-8") as f:
        json.dump(failure_gallery, f, indent=2)

    gallery_lines = ["# Failure Gallery"]
    gallery_lines.append(f"- run_id: `{run_id}`")
    gallery_lines.append(f"- dataset: `{dataset_path}`")
    gallery_lines.append("")
    gallery_lines.append("## Failure counts")
    for k in sorted(failure_counts.keys()):
        desc = failure_taxonomy.get(k, "")
        suffix = f" — {desc}" if desc else ""
        gallery_lines.append(f"- **{k}**: {failure_counts[k]}{suffix}")
    gallery_lines.append("")
    gallery_lines.append("## Failures")
    for fitem in failures:
        gallery_lines.append(
            f"- `{fitem['id']}` [{fitem['category']}] {fitem['failure_type']} :: {fitem['answer_preview']}"
        )
    with open(os.path.join(run_dir, "failure_gallery.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(gallery_lines))

    print(f"[eval] wrote {run_dir}/metrics.json")
    print(f"[eval] wrote {run_dir}/report.md")
    print(f"[eval] wrote {run_dir}/failures.json")
    print(f"[eval] wrote {run_dir}/failure_gallery.json")
    print(f"[eval] wrote {run_dir}/failure_gallery.md")
    print(f"[eval] wrote {run_dir}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
