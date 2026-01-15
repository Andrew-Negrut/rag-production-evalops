# # import re
# # from typing import List

# # def _snap_left_to_word_boundary(text: str, idx: int) -> int:
# #     """Move idx left if it's in the middle of a word."""
# #     if idx <= 0:
# #         return 0
# #     idx = min(idx, len(text))
# #     while idx > 0 and idx < len(text) and text[idx - 1].isalnum() and text[idx].isalnum():
# #         idx -= 1
# #     return idx

# # def _snap_right_to_word_boundary(text: str, idx: int) -> int:
# #     """Move idx right if it's in the middle of a word."""
# #     idx = max(0, min(idx, len(text)))
# #     while idx > 0 and idx < len(text) and text[idx - 1].isalnum() and text[idx].isalnum():
# #         idx += 1
# #     return idx

# # def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
# #     text = text.strip()
# #     if not text:
# #         return []

# #     # Normalize newlines
# #     text = text.replace("\r\n", "\n").replace("\r", "\n")

# #     # Split into paragraphs (blank-line separated)
# #     paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
# #     if not paras:
# #         return []

# #     chunks: List[str] = []
# #     cur: List[str] = []
# #     cur_len = 0

# #     def flush():
# #         nonlocal cur, cur_len
# #         if cur:
# #             chunks.append("\n\n".join(cur).strip())
# #         cur = []
# #         cur_len = 0

# #     for p in paras:
# #         p_len = len(p)
# #         # If a single paragraph is huge, fall back to hard slicing (rare)
# #         if p_len > chunk_size:
# #             flush()
# #             start = 0
# #             while start < p_len:
# #                 end = min(start + chunk_size, p_len)
# #                 chunks.append(p[start:end].strip())
# #                 if end == p_len:
# #                     break
# #                 start = max(0, end - overlap)
# #             continue

# #         # If adding this paragraph would exceed chunk_size, flush current chunk
# #         if cur_len and (cur_len + 2 + p_len) > chunk_size:
# #             flush()

# #         cur.append(p)
# #         cur_len += (2 if cur_len else 0) + p_len

# #     flush()

# #     # Optional: lightweight overlap by carrying tail paragraphs forward
# #     if overlap > 0 and len(chunks) > 1:
# #         overlapped: List[str] = [chunks[0]]
# #         for i in range(1, len(chunks)):
# #             prev = chunks[i - 1]
# #             tail = prev[-overlap:].strip()
# #             merged = (tail + "\n\n" + chunks[i]).strip() if tail else chunks[i]
# #             overlapped.append(merged)
# #         chunks = overlapped

# #     return chunks

# from typing import List, Dict, Optional
# import re

# _HEADING_RE = re.compile(r"^[A-Z][A-Z0-9&'’\-\s]{2,80}$")

# def _is_heading(line: str) -> bool:
#     line = line.strip()
#     if len(line) < 4 or len(line) > 80:
#         return False
#     # Common in PDFs: ALL CAPS headings like "FUND SUMMARY", "FEES AND EXPENSES"
#     if _HEADING_RE.match(line) and line.upper() == line:
#         return True
#     return False

# def chunk_text(
#     text: str,
#     chunk_size: int = 800,
#     overlap: int = 120
# ) -> List[Dict[str, Optional[str]]]:
#     """
#     Returns: [{"text": "...", "section": "..."}, ...]
#     """
#     text = text.strip()
#     if not text:
#         return []

#     lines = [ln.rstrip() for ln in text.splitlines()]
#     chunks: List[Dict[str, Optional[str]]] = []

#     section: Optional[str] = None
#     buf: str = ""

#     def flush():
#         nonlocal buf
#         t = buf.strip()
#         if t:
#             chunks.append({"text": t, "section": section})
#         buf = ""

#     for ln in lines:
#         s = ln.strip()
#         if not s:
#             continue

#         if _is_heading(s):
#             # New section begins → flush current buffer first
#             flush()
#             section = s.title()
#             continue

#         # accumulate
#         if buf:
#             buf += "\n"
#         buf += s

#         if len(buf) >= chunk_size:
#             flush()
#             # start next buffer with overlap tail
#             buf = buf[-overlap:]

#     flush()
#     return chunks

from __future__ import annotations

from typing import List, Dict, Tuple
import re

# Small, finance-doc-structure-aware heading set (not Vanguard-specific)
# IMPORTANT: patterns match headings at start of line, but allow trailing junk like:
# "Investment Objective .......... 1" or "Investment Objective (continued)"
_HEADING_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Fund Summary", re.compile(r"^\s*Fund\s+Summary\b.*$", re.I)),
    ("Investment Objective", re.compile(r"^\s*Investment\s+Objective\b.*$", re.I)),
    ("Fees and Expenses", re.compile(r"^\s*Fees\s+and\s+Expenses\b.*$", re.I)),
    ("Principal Risks", re.compile(r"^\s*(Principal\s+Risks|Risks)\b.*$", re.I)),
    ("Investment Strategies", re.compile(r"^\s*(Investment\s+Strategies|Strategies)\b.*$", re.I)),
    ("Performance", re.compile(r"^\s*Performance\b.*$", re.I)),
    ("Portfolio Turnover", re.compile(r"^\s*Portfolio\s+Turnover\b.*$", re.I)),
    ("Glossary", re.compile(r"^\s*Glossary(\s+of\s+Investment\s+Terms)?\b.*$", re.I)),
]

def _build_section_breakpoints(text: str) -> List[Tuple[int, str]]:
    """
    Returns list of (char_offset, section_name). We update section when we see a heading line.
    """
    bps: List[Tuple[int, str]] = [(0, "General")]
    offset = 0
    current = "General"

    for line in text.splitlines(True):  # keep newline to maintain offsets
        stripped = line.strip()

        # Avoid matching long paragraph lines as headings
        # (PDF extract_text sometimes produces long runs of text in a single line)
        if stripped and len(stripped) <= 120:
            for name, pat in _HEADING_PATTERNS:
                if pat.match(stripped):
                    current = name
                    bps.append((offset, current))
                    break

        offset += len(line)

    return bps

def _section_at(bps: List[Tuple[int, str]], pos: int) -> str:
    # last breakpoint <= pos
    sec = bps[0][1]
    for off, name in bps:
        if off <= pos:
            sec = name
        else:
            break
    return sec

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, str]]:
    """
    Chunk a single text blob into [{"text": ..., "section": ...}, ...]
    """
    text = (text or "").strip()
    if not text:
        return []

    bps = _build_section_breakpoints(text)

    chunks: List[Dict[str, str]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        section = _section_at(bps, start)

        chunks.append({"text": chunk, "section": section})

        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks

def chunk_pages(pages: List[str], chunk_size: int = 800, overlap: int = 100) -> List[Dict[str, object]]:
    """
    Option B: chunk page-by-page and attach page number + section.

    Returns list like:
      [{"text": "...", "section": "...", "page": 1}, ...]
    """
    chunks: List[Dict[str, object]] = []
    current_section = "General"

    for page_num, page_text in enumerate(pages or [], start=1):
        page_text = (page_text or "").strip()
        if not page_text:
            continue

        bps = _build_section_breakpoints(page_text)

        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk = page_text[start:end]

            sec = _section_at(bps, start)

            # carry-forward section if the page doesn't explicitly restate a heading
            if sec == "General":
                sec = current_section
            else:
                current_section = sec

            chunks.append(
                {"text": chunk, "section": sec, "page": page_num}
            )

            if end == len(page_text):
                break
            start = max(0, end - overlap)

    return chunks
