import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI

app = FastAPI(title="SAP RAG Backend")

# ---------- ENV ----------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_KEY)

# ---------- MODELS ----------
class DocumentIn(BaseModel):
    title: str
    content: str
    source: str | None = None

# ---------- HEALTH ----------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- CHUNK HELPER ----------
def chunk_text(text: str, size: int = 800, overlap: int = 100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks

# ---------- EMBEDDING ----------
def embed_text(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ---------- INGEST ----------
@app.post("/ingest")
async def ingest_document(doc: DocumentIn):

    # insert document metadata
    doc_id = str(uuid.uuid4())

    supabase.table("documents").insert({
        "id": doc_id,
        "title": doc.title,
        "content": doc.content,
        "source": doc.source
    }).execute()

    # chunk + embed
    chunks = chunk_text(doc.content)

    for chunk in chunks:
        embedding = embed_text(chunk)

        supabase.table("document_embeddings").insert({
            "document_id": doc_id,
            "chunk": chunk,
            "embedding": embedding
        }).execute()

    return {
        "status": "success",
        "chunks": len(chunks),
        "document_id": doc_id
    }
