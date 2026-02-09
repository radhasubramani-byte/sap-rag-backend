import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from supabase import create_client
from openai import OpenAI

# =========================
# Environment setup
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="SAP RAG Backend")

# =========================
# Models
# =========================

class DocumentIn(BaseModel):
    title: str
    content: str
    source: str | None = None


class AskRequest(BaseModel):
    question: str
    user_email: str | None = None


# =========================
# Helpers
# =========================

def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(" ".join(current)) > chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def embed_text(text: str):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


# =========================
# Health endpoint
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Ingest endpoint
# =========================

@app.post("/ingest")
async def ingest_document(doc: DocumentIn):

    # Save document metadata
    doc_result = supabase.table("documents").insert({
        "title": doc.title,
        "content": doc.content,
        "source": doc.source
    }).execute()

    document_id = doc_result.data[0]["id"]

    # Chunk content
    chunks = chunk_text(doc.content)

    # Embed + store chunks
    for chunk in chunks:
        embedding = embed_text(chunk)

        supabase.table("document_embeddings").insert({
            "document_id": document_id,
            "chunk": chunk,
            "embedding": embedding
        }).execute()

    return {
        "status": "success",
        "chunks": len(chunks),
        "document_id": document_id
    }


# =========================
# Ask / RAG endpoint
# =========================

@app.post("/ask")
async def ask_rag(req: AskRequest):
    try:
        # Embed user question
        query_embedding = embed_text(req.question)

        # Vector search via Supabase RPC
        result = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 0.5,
                "match_count": 5
            }
        ).execu
