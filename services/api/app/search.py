import re
from typing import List, Tuple

_WORD = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())

def keyword_score(query: str, text: str) -> int:
    q = set(tokenize(query))
    if not q:
        return 0
    t = tokenize(text)
    return sum(1 for w in t if w in q)

def rank_chunks(query: str, chunks: List[Tuple[str, str, int, str]], top_k: int = 5):
    """
    chunks: list of (chunk_id, document_id, chunk_index, content)
    """
    scored = []
    for chunk_id, doc_id, idx, content in chunks:
        s = keyword_score(query, content)
        if s > 0:
            scored.append((s, chunk_id, doc_id, idx, content))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]
