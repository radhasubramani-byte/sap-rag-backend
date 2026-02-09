import os
import math
import ast
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI

# =============================
# Environment setup
# =============================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY]):
    raise Exception("Missing environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# FastAPI app
# =============================

app = FastAPI(title="SAP RAG Backend")

# =============================
# Models
# =============================

class DocumentIn(BaseModel):
    title: str
    content: str
    source: str

class QuestionIn(BaseModel):
    question: str
    user_email: str | None = None

# =============================
# Helpers
# =============================

def chunk_text(text: str, size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current = []

    for word in words:
        current.append(word)
        if len(current) >= size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def embed_text(text: str) -> List[float]:
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-9)


# =============================
# Health endpoint
# =============================

@app.get("/health")
async def health():
    return {"status": "ok"}


# =============================
# Ingestion endpoint
# =============================

@app.post("/ingest")
async def ingest_document(doc: DocumentIn):
    try:
        # store document
        doc_insert = supabase.table("documents").insert({
            "title": doc.title,
            "content": doc.content,
            "source": doc.source
        }).execute()

        document_id = doc_insert.data[0]["id"]

        # chunk + embed
        chunks = chunk_text(doc.content)

        for chunk in chunks:
            embedding = embed_text(chunk)

            supabase.table("document_embeddings").insert({
                "document_id": document_id,
                "embedding": embedding,
                "chunk": chunk
            }).execute()

        return {
            "status": "success",
            "chunks": len(chunks),
            "document_id": document_id
        }

    except Exception as e:
        return {"error": str(e)}


# =============================
# RAG Ask endpoint
# =============================

@app.post("/ask")
async def ask_rag(q: QuestionIn):
    try:
        # embed question
        query_embedding = embed_text(q.question)

        rows = supabase.table("document_embeddings").select("*").execute().data

        if not rows:
            return {
                "answer": "No knowledge found.",
                "confidence": "low"
            }

        best_score = -1
        best_chunk = ""

        for row in rows:
            vec = row["embedding"]

            # 🔥 SAFE VECTOR PARSE
            if isinstance(vec, str):
                vec = ast.literal_eval(vec)

            score = cosine_similarity(query_embedding, vec)

            if score > best_score:
                best_score = score
                best_chunk = row["chunk"]

        # LLM answer
        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an SAP assistant. Answer using provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{best_chunk}\n\nQuestion:\n{q.question}"
                }
            ]
        )

        answer = completion.choices[0].message.content

        confidence = (
            "high" if best_score > 0.85
            else "medium" if best_score > 0.65
            else "low"
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "similarity": round(best_score, 3)
        }

    except Exception as e:
        return {"detail": str(e)}
