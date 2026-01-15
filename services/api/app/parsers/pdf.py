from __future__ import annotations

import io
from typing import List

from pypdf import PdfReader


class PdfExtractError(RuntimeError):
    pass


def extract_pages_from_pdf_bytes(data: bytes, max_pages: int | None = None) -> List[str]:
    """
    Extract text from a PDF file as a list of per-page strings.

    Notes:
    - PDF text extraction is imperfect. Some PDFs are scanned images and will return empty text
      unless you add OCR later.
    """
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise PdfExtractError(f"Failed to read PDF: {e}") from e

    pages = reader.pages[:max_pages] if max_pages else reader.pages
    out: List[str] = []

    for page in pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""

        t = (t or "").strip()
        out.append(t)

    return out


def extract_text_from_pdf_bytes(data: bytes, max_pages: int | None = None) -> str:
    """
    Backward-compatible: Extract text from a PDF and join pages with blank lines.
    """
    pages = extract_pages_from_pdf_bytes(data, max_pages=max_pages)
    texts = [p for p in pages if p and p.strip()]
    return "\n\n".join(texts).strip()
