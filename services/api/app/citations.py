# from __future__ import annotations

# import re
# from typing import Dict, List, Tuple, Any


# _CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# def extract_citation_numbers(text: str) -> List[int]:
#     nums: List[int] = []
#     for m in _CITE_RE.finditer(text):
#         part = m.group(1)
#         for raw in part.split(","):
#             raw = raw.strip()
#             if raw.isdigit():
#                 nums.append(int(raw))
#     # stable unique order
#     seen = set()
#     out = []
#     for n in nums:
#         if n not in seen:
#             seen.add(n)
#             out.append(n)
#     return out


# def build_sources_block(results: List[Dict[str, Any]], max_chars_per_source: int = 900) -> Tuple[str, Dict[int, Dict[str, Any]]]:
#     """
#     Returns:
#       sources_text: string inserted into prompt
#       idx_map: 1-based index -> chunk metadata
#     """
#     idx_map: Dict[int, Dict[str, Any]] = {}
#     lines: List[str] = ["SOURCES:"]
#     for i, r in enumerate(results, start=1):
#         idx_map[i] = r
#         content = (r.get("content_preview") or "").strip()
#         content = content[:max_chars_per_source]
#         lines.append(
#             f"[{i}] document_id={r.get('document_id')} chunk_id={r.get('chunk_id')} chunk_index={r.get('chunk_index')}\n{content}"
#         )
#     return "\n\n".join(lines), idx_map

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citation_numbers(text: str) -> List[int]:
    nums: List[int] = []
    for m in _CITE_RE.finditer(text):
        part = m.group(1)
        for raw in part.split(","):
            raw = raw.strip()
            if raw.isdigit():
                nums.append(int(raw))

    # stable unique order
    seen = set()
    out: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def build_sources_block(
    results: List[Dict[str, Any]],
    max_chars_per_source: int = 900,
) -> Tuple[str, Dict[int, Dict[str, Any]]]:
    """
    Returns:
      sources_text: string inserted into prompt
      idx_map: 1-based index -> chunk metadata (the same dict from results)
    """
    idx_map: Dict[int, Dict[str, Any]] = {}
    lines: List[str] = ["SOURCES:"]

    for i, r in enumerate(results, start=1):
        idx_map[i] = r

        # Prefer full content if present, else fallback to preview.
        content = (r.get("content") or r.get("content_preview") or "").strip()
        if max_chars_per_source and max_chars_per_source > 0:
            content = content[:max_chars_per_source]

        lines.append(
            f"[{i}] document_id={r.get('document_id')} chunk_id={r.get('chunk_id')} chunk_index={r.get('chunk_index')}\n{content}"
        )

    return "\n\n".join(lines), idx_map
