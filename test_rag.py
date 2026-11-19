#!/usr/bin/env python3
"""
Test script for the RAG pipeline (without LLM call).
Run: python test_rag.py "ваш запрос"
"""
import sys
import json
from rag_engine import MultiBookRAG

def test_query(query: str, mode: str = "fts", top_k: int = 5):
    rag = MultiBookRAG()
    results = rag.search_all(query, top_k=top_k, mode=mode)
    
    print(f"\n{'='*60}")
    print(f"Запрос: {query}")
    print(f"Режим: {mode}, top_k: {top_k}")
    print(f"Найдено: {len(results)} фрагментов")
    print(f"{'='*60}\n")
    
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['book_title']}")
        print(f"    Глава: {r['chapter'] or '?'} | PDF-с. {r['pdf_page']} | Часть: {r['part'] or '?'}")
        print(f"    Score: {r['fused']:.5f}")
        print(f"    Текст: {r['text'][:200]}...")
        print()
    
    # Test context formatting
    context = rag.format_context(results)
    print(f"{'='*60}")
    print("КОНТЕКСТ ДЛЯ LLM:")
    print(f"{'='*60}")
    print(context[:2000])
    print("..." if len(context) > 2000 else "")
    
    return results

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "лактатный порог"
    mode = sys.argv[2] if len(sys.argv) > 2 else "fts"
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    test_query(query, mode, top_k)