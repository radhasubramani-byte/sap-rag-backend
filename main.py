from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    user_email: str | None = None

@app.post("/ask")
def ask(payload: QuestionRequest):
    return {
        "answer": (
            "This is a test response from FastAPI.\n\n"
            "Your question was:\n"
            f"{payload.question}\n\n"
            "Next step: connect real SAP RAG logic."
        ),
        "confidence": 0.1
    }