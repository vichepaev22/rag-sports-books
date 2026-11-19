#!/usr/bin/env python3
"""
FastAPI server for the RAG engine.
Deployable on Railway (free tier).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn

from rag_engine import MultiBookRAG

app = FastAPI(
    title="Sports Physiology RAG API",
    description="RAG over Wilmore/Costill + Issurin books with Nemotron-3-Ultra",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = MultiBookRAG()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Search query in Russian")
    top_k: int = Field(5, ge=1, le=10)
    chapter: Optional[int] = Field(None, ge=1, le=22, description="Filter by chapter number")
    mode: str = Field("hybrid", pattern="^(fts|vector|hybrid)$")


class SourceResponse(BaseModel):
    skill: str
    book: str
    chapter: Optional[int]
    pdf_page: int
    part: Optional[str]
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceResponse]


@app.get("/")
async def root():
    return {
        "service": "Sports Physiology RAG",
        "books": [
            "Уилмор и Костилл — Физиология спорта и двигательной активности (459 стр.)",
            "Иссурин — Подготовка спортсменов XXI века (460 стр.)",
        ],
        "endpoints": {
            "POST /query": "Ask a question",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    try:
        result = rag.answer(req.query, req.top_k, req.chapter, req.mode)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)