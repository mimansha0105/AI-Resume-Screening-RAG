"""Minimal FastAPI endpoint for resume retrieval."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .pipeline import ResumeRAG

app = FastAPI(title="AI Resume Screening RAG API", version="1.0.0")
rag = ResumeRAG()

class ScreenRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": rag.collection.count()}

@app.post("/screen")
def screen(request: ScreenRequest):
    try:
        return rag.build_generation_context(request.query, request.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
