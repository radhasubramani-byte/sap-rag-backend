import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI
import tiktoken

# ---------------------------------------------------
# ENVIRONMENT CHECK
# ---------------------------------------------------

required_env = [
    "OPENAI_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY"
]

for key in required_env:
    if not os.getenv(key):
        raise RuntimeError(f"Missing environment variable: {key}")

# ---------------------------------------------------
# CLIENT INITIALIZATION
# ---------------------------------------------------

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI(title="SAP RAG Backend")

# ---------------------------------------------------
# TOKENIZER FOR CHUNKING
# ---------------------------------------------------

tokenizer = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 700):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------
# EMBEDDING HELPER
# ---------------------------------------------------

def embed_text(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ---------------------------------------------------
# REQUEST MODEL
# ---------------------------------------------------

class IngestRequest(BaseModel):
    title: str
    content: str
    source: str


# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------

@app.get("/")
def health():
    return {"status": "SAP RAG backend running"}


# ---------------------------------------------------
#
