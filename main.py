from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LegalQuestion(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Legal Assistant Backend is running!"}

@app.post("/ask")
def ask_legal_question(data: LegalQuestion):
    answer, sources = get_answer(data.question)
    return {
        "question": data.question,
        "answer": answer,
        "sources": sources,
    }