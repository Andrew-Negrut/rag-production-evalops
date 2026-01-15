from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


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


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = Request(url=url, method=method.upper(), data=data, headers=headers)
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} calling {url}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Failed to call {url}: {e}") from e


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


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m services.api.eval.run_route1 eval/golden.route1.jsonl [base_url]")
        return 2

    dataset_path = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_BASE_URL

    examples = _load_jsonl(dataset_path)

    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    failures: List[Dict[str, Any]] = []

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

    # simple markdown report
    report_lines = []
    report_lines.append(f"# Route 1 Eval Report\n")
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

    print(f"[eval] wrote {run_dir}/metrics.json")
    print(f"[eval] wrote {run_dir}/report.md")
    print(f"[eval] wrote {run_dir}/failures.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
