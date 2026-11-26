"""
This module is used to rerank retrieved chunks (dictionaries with text, source,
and page metadata) using an open-source cross-encoder model.

Usage:
    from rag.reranker import get_reranker, rerank

    reranker_model = get_reranker()
    chunks_for_context = rerank(reranker_model, "Kitob qachon yozilgan?", chunks, top_k=7)
"""

from FlagEmbedding import FlagReranker
from functools import lru_cache
from app.config import settings

@lru_cache(maxsize=1)
def get_reranker() -> FlagReranker:
    return FlagReranker(model_name_or_path=settings.reranker_model)

def rerank(model: FlagReranker, 
           query: str, 
           chunks: list[dict], 
           top_k: int = 7) -> list[dict]:
    if not chunks:
        return []
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = model.compute_score(sentence_pairs=pairs, normalize=True)

    if isinstance(scores, float):
        scores = [scores]

    scored = sorted(zip(scores, chunks),
                    key=lambda x: x[0], 
                    reverse=True)[:top_k]

    return [{**chunk, "rerank_score": round(float(score), 4)} for score, chunk in 
            scored]