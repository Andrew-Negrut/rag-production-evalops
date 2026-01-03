import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import select

from db import Base, engine, SessionLocal
from models import Document

app = FastAPI()

Base.metadata.create_all(bind=engine)

class IngestRequest(BaseModel):
    title: str
    content: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/documents")
def create_document(req: IngestRequest):
    doc_id = str(uuid.uuid4())
    with SessionLocal() as db:
        db.add(Document(id=doc_id, title=req.title, content=req.content))
        db.commit()
    return {"id": doc_id}

@app.get("/documents")
def list_documents():
    with SessionLocal() as db:
        rows = db.execute(select(Document.id, Document.title)).all()
    return [{"id": r[0], "title": r[1]} for r in rows]
