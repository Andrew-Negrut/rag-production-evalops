import os
from fastapi.testclient import TestClient

os.environ["ENV"] = "dev"
os.environ["LLM_PROVIDER"] = "fake"

from app.main import app


def _make_minimal_pdf_with_text(text: str) -> bytes:
    """
    Create a tiny valid PDF containing one line of text.
    This avoids adding heavy PDF-generation dependencies.
    """
    # Basic PDF with Helvetica and one content stream.
    # We must build a correct xref with byte offsets.
    objects = []

    def obj(n: int, body: str) -> bytes:
        return f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")

    # 1: Catalog
    objects.append(obj(1, "<< /Type /Catalog /Pages 2 0 R >>"))
    # 2: Pages
    objects.append(obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"))
    # 3: Page
    objects.append(
        obj(
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        )
    )

    # 4: Contents stream
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 24 Tf 50 100 Td ({safe}) Tj ET"
    stream_bytes = stream.encode("latin-1")
    objects.append(
        b"4 0 obj\n<< /Length %d >>\nstream\n%s\nendstream\nendobj\n"
        % (len(stream_bytes), stream_bytes)
    )

    # 5: Font
    objects.append(obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

    # Build file with offsets
    offsets = []
    out = bytearray()
    out += header
    offsets.append(0)  # object 0 is special; will be written in xref as free

    # write objects 1..5
    for i, ob in enumerate(objects, start=1):
        offsets.append(len(out))
        out += ob

    xref_start = len(out)
    out += b"xref\n0 6\n"
    out += b"0000000000 65535 f \n"
    for i in range(1, 6):
        out += f"{offsets[i]:010d} 00000 n \n".encode("latin-1")

    out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
    out += f"{xref_start}\n".encode("latin-1")
    out += b"%%EOF\n"
    return bytes(out)


def test_upload_pdf_ingests_document():
    with TestClient(app) as c:
        r = c.post("/dev/reset")
        assert r.status_code == 200

        pdf_bytes = _make_minimal_pdf_with_text("Hello PDF world. Refunds allowed within 30 days.")
        files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}
        data = {"title": "Sample Prospectus"}

        r = c.post("/documents/upload_pdf", data=data, files=files)
        assert r.status_code == 200, r.text
        resp = r.json()
        assert "id" in resp
        assert resp["chunks_created"] > 0

        # sanity: list documents includes it
        r = c.get("/documents")
        assert r.status_code == 200
        docs = r.json()
        assert any(d["title"] == "Sample Prospectus" for d in docs)
