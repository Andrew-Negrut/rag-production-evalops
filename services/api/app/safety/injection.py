from __future__ import annotations

import re
from typing import Optional


_PATTERNS = [
    re.compile(r"\b(ignore|override|bypass)\b.*\b(instructions|rules|system)\b", re.I),
    re.compile(r"\b(reveal|show|dump)\b.*\b(system prompt|prompt|instructions)\b", re.I),
    re.compile(r"\b(list|exfiltrate|leak)\b.*\b(documents|database|secrets|keys)\b", re.I),
    re.compile(r"\btools?\b.*\b(list|schema|schemas|available)\b", re.I),
    re.compile(r"\b(list|show|explain)\b.*\btools?\b", re.I),
    re.compile(r"\b(system prompt|system instruction|system message)\b", re.I),
    re.compile(r"\b(list|dump|export)\b.*\b(documents?|database|contents?)\b", re.I),
    re.compile(r"\b(api key|secret|secrets|credentials|token)\b", re.I),
]


def detect_injection(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None
    for pat in _PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None
