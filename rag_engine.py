#!/usr/bin/env python3
"""
RAG engine for two sports physiology books.
Combines FTS5 + LSA vector search from both skills, generates answers via Nemotron-3-Ultra.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Optional ML dependencies
try:
    import numpy as np
    import joblib
    HAS_ML = True
except ImportError:
    HAS_ML = False
    np = None
    joblib = None

# Skill paths
SKILLS_ROOT = Path("/home/hermes/.hermes/skills/books")
SKILL_CONFIG = {
    "physiology-sport-skill": {
        "db": "sport_physiology.sqlite",
        "title": "Уилмор и Костилл — Физиология спорта и двигательной активности",
        "vectorizer": "vectorizer.joblib",
        "vectors": "vectors_lsa128.npz",
    },
    "issurin-training-skill": {
        "db": "issurin_training.sqlite",
        "title": "Иссурин — Подготовка спортсменов XXI века, 2016",
        "vectorizer": "vectorizer.joblib",
        "vectors": "vectors_lsa128.npz",
    },
}


class BookSearcher:
    """Searches a single skill's SQLite FTS5 + LSA vectors."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        self.config = SKILL_CONFIG[skill_name]
        self.data_dir = SKILLS_ROOT / skill_name / "data"
        self.db_path = self.data_dir / self.config["db"]
        self._vectorizer = None
        self._vectors = None
        self._ids = None
        self._chapters = None

    def _load_vectors(self):
        if not HAS_ML:
            raise RuntimeError("ML dependencies (numpy, joblib) not available. Install them for vector search.")
        if self._vectors is None:
            pack = joblib.load(self.data_dir / self.config["vectorizer"])
            vals = np.load(self.data_dir / self.config["vectors"])
            self._vectorizer = pack["vectorizer"]
            self._svd = pack["svd"]
            self._vectors = vals["vectors"]
            self._ids = vals["chunk_ids"].astype(int)
            self._chapters = vals["chapters"].astype(int)

    def search(
        self,
        query: str,
        top_k: int = 5,
        chapter: Optional[int] = None,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        candidates = {}

        # FTS5 search
        if mode in ("fts", "hybrid"):
            terms = re.findall(r"(?u)[а-яёa-z0-9]{3,}", query.lower())[:16]
            if terms:
                match = " OR ".join(f'"{t}"*' for t in terms)
                sql = """
                    SELECT c.id, c.pdf_page, c.chapter, c.part, c.content,
                           bm25(chunks_fts) as bm
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                """
                params = [match]
                if chapter is not None:
                    sql += " AND c.chapter = ?"
                    params.append(chapter)
                sql += " ORDER BY bm LIMIT 80"

                with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as db:
                    db.row_factory = sqlite3.Row
                    rows = db.execute(sql, params).fetchall()
                    for rank, r in enumerate(rows, 1):
                        candidates[r["id"]] = {
                            "id": r["id"],
                            "pdf_page": r["pdf_page"],
                            "chapter": r["chapter"],
                            "part": r["part"],
                            "text": r["content"],
                            "fused": 1 / (60 + rank),
                            "bm25": round(float(r["bm"]), 4),
                            "skill": self.skill_name,
                            "book_title": self.config["title"],
                        }

                # Fallback to OR if no results
                if not rows and len(terms) > 1:
                    match_or = " OR ".join(f'"{t}"' for t in terms)
                    sql_or = sql.replace(match, match_or)
                    rows = db.execute(sql_or, params).fetchall()
                    for rank, r in enumerate(rows, 1):
                        candidates[r["id"]] = {
                            "id": r["id"],
                            "pdf_page": r["pdf_page"],
                            "chapter": r["chapter"],
                            "part": r["part"],
                            "text": r["content"],
                            "fused": 1 / (60 + rank),
                            "bm25": round(float(r["bm"]), 4),
                            "skill": self.skill_name,
                            "book_title": self.config["title"],
                        }

        # Vector search (LSA)
        if mode in ("vector", "hybrid") and HAS_ML:
            try:
                self._load_vectors()
                vec = self._svd.transform(self._vectorizer.transform([query]))[0]
                mag = np.linalg.norm(vec)
                if mag > 0:
                    scores = self._vectors @ (vec / mag)
                    if chapter is not None:
                        available = np.flatnonzero(self._chapters == chapter)
                    else:
                        available = np.arange(len(scores))
                    if len(available) > 0:
                        keep_n = min(80, len(available))
                        ix = available[np.argsort(scores[available])[-keep_n:][::-1]]
                        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as db:
                            db.row_factory = sqlite3.Row
                            for rank, i in enumerate(ix, 1):
                                ident = int(self._ids[i])
                                score = float(scores[i])
                                if ident not in candidates:
                                    r = db.execute(
                                        "SELECT id, pdf_page, chapter, part, content FROM chunks WHERE id = ?",
                                        (ident,),
                                    ).fetchone()
                                    if r:
                                        candidates[ident] = {
                                            "id": r["id"],
                                            "pdf_page": r["pdf_page"],
                                            "chapter": r["chapter"],
                                            "part": r["part"],
                                            "text": r["content"],
                                            "fused": 0.0,
                                            "skill": self.skill_name,
                                            "book_title": self.config["title"],
                                        }
                                candidates[ident]["fused"] += 1 / (60 + rank)
                                candidates[ident]["cosine_lsa"] = round(score, 4)
            except Exception as e:
                # Vector search failed, continue with FTS only
                pass

        # Sort by fused score
        results = sorted(candidates.values(), key=lambda x: x["fused"], reverse=True)[:top_k]
        return results


class MultiBookRAG:
    """RAG over both skills combined."""

    def __init__(self):
        self.searchers = {
            name: BookSearcher(name) for name in SKILL_CONFIG.keys()
        }

    def search_all(
        self,
        query: str,
        top_k: int = 5,
        chapter: Optional[int] = None,
        mode: str = "hybrid",
    ) -> List[Dict[str, Any]]:
        all_results = []
        for searcher in self.searchers.values():
            results = searcher.search(query, top_k, chapter, mode)
            all_results.extend(results)
        # Re-rank globally
        all_results.sort(key=lambda x: x["fused"], reverse=True)
        return all_results[:top_k]

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for LLM prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Источник {i}] {r['book_title']} | Глава {r['chapter'] or '?'} | PDF-с. {r['pdf_page']} | Часть {r['part'] or '?'}\n{r['text']}"
            )
        return "\n\n".join(parts)

    def build_prompt(self, query: str, context: str) -> str:
        return f"""Ты — эксперт по спортивной физиологии и методике тренировок. Отвечай на русском.

Контекст из двух книг:
{context}

Правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста.
2. Обязательно цитируй источники: [Источник 1], [Источник 2] и т.д.
3. Указывай PDF-страницы и главы из цитируемых источников.
4. Если в контексте нет ответа — прямо скажи: "В предоставленных фрагментах ответа нет."
5. Разделяй факты книги от собственных выводов пометкой "(адаптация)".

Вопрос: {query}

Ответ:"""

    def generate_answer(self, query: str, context: str) -> str:
        """Call Nemotron-3-Ultra via NVIDIA API."""
        import os
        import aiohttp
        import asyncio

        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            return "❌ NVIDIA_API_KEY не задан"

        prompt = self.build_prompt(query, context)

        async def call_llm():
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "messages": [
                    {"role": "system", "content": "Ты — эксперт по спортивной физиологии и методике тренировок. Отвечай на русском с цитированием источников."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 2000,
                "temperature": 0.2,
                "stream": False,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        return f"❌ LLM error {resp.status}: {err}"
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

        return asyncio.run(call_llm())

    def answer(self, query: str, top_k: int = 5, chapter: Optional[int] = None, mode: str = "hybrid") -> Dict[str, Any]:
        """Full RAG pipeline: search → format → generate."""
        results = self.search_all(query, top_k, chapter, mode)
        if not results:
            return {
                "query": query,
                "answer": "Ничего не найдено по запросу.",
                "sources": [],
            }
        context = self.format_context(results)
        answer = self.generate_answer(query, context)
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "skill": r["skill"],
                    "book": r["book_title"],
                    "chapter": r["chapter"],
                    "pdf_page": r["pdf_page"],
                    "part": r["part"],
                    "score": r["fused"],
                }
                for r in results
            ],
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG over two sports physiology books")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chapter", type=int, default=None)
    parser.add_argument("--mode", choices=["fts", "vector", "hybrid"], default="hybrid")
    args = parser.parse_args()

    rag = MultiBookRAG()
    result = rag.answer(args.query, args.top_k, args.chapter, args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))