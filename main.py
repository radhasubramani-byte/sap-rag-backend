import os
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI

# ---------------------------------------------------
# Environment setup
# ---------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY]):
    raise RuntimeError("Missing environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------

app = FastAPI(title="SAP RAG Backend")

# ---------------------------------------------------
# Models
# ---------------------------------------------------

class AskRequest(BaseModel):
    question: str
    user_email: str | None = None


class DocumentIn(BaseModel):
    title: str
    content: str
    source: str | None = None

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


def embed_text(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# ---------------------------------------------------
# Health endpoint
# ---------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------
# Ingestion endpoint
# ---------------------------------------------------

@app.post("/ingest")
async def ingest_document(doc: DocumentIn):
    try:
        # Save document
        doc_result = supabase.table("documents").insert({
            "title": doc.title,
            "content": doc.content,
            "source": doc.source
        }).execute()

        document_id = doc_result.data[0]["id"]

        # Chunk text
        chunks = chunk_text(doc.content)

        # Embed + store
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
        print("INGEST ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------
# Ask endpoint (RAG retrieval)
# ---------------------------------------------------

@app.post("/ask")
async def ask_rag(req: AskRequest):
    try:
        question_embedding = embed_text(req.question)

        embeddings_data = supabase.table("document_embeddings") \
            .select("*") \
            .limit(200) \
            .execute()

        scored_chunks = []

        for row in embeddings_data.data:
            score = cosine_similarity(question_embedding, row["embedding"])
            scored_chunks.append((score, row["chunk"]))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        top_context = "\n\n".join(chunk for _, chunk in scored_chunks[:5])

        prompt = f"""
Use the context below to answer the question.

Context:
{top_context}

Question:
{req.question}
"""

        completion = openai_client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "confidence": "medium"
        }

    except Exception as e:
        print("ASK ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
