"""Vector store, embedding, and intent-boosted retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.models import (
    CTA_KEYWORDS, QUESTION_SET, VALUE_PROP_KEYWORDS, ContentChunk,
)

# ── Embedding + FAISS ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class VectorStore:
    index: faiss.IndexFlatIP
    chunks: list[ContentChunk]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_vector_store(chunks: list[ContentChunk]) -> VectorStore:
    model = get_embedding_model()
    vectors = model.encode([c.text for c in chunks], convert_to_numpy=True)
    vectors = _normalize(vectors.astype(np.float32))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return VectorStore(index=index, chunks=chunks)


def _search(store: VectorStore, query: str, top_k: int) -> tuple[list[int], list[float]]:
    model = get_embedding_model()
    qvec = _normalize(model.encode([query], convert_to_numpy=True).astype(np.float32))
    scores, ids = store.index.search(qvec, top_k)
    return ids[0].tolist(), scores[0].tolist()

# ── Intent boosting + deduplication ──────────────────────────────────────────

def _boost_score(chunk: ContentChunk, base_score: float, question_key: str) -> float:
    text = chunk.text.lower()
    boost = 0.0
    if question_key == "business_value":
        if chunk.metadata.page_type == "homepage":
            boost += 0.20
        if any(k in text for k in VALUE_PROP_KEYWORDS):
            boost += 0.15
    if question_key == "cta":
        if any(k in text for k in CTA_KEYWORDS):
            boost += 0.30
    if question_key in {"audience", "trust_gaps"}:
        for key in ("pricing", "contact", "service", "services"):
            if key in text:
                boost += 0.08
    return base_score + boost


def _dedupe(chunks: list[ContentChunk]) -> list[ContentChunk]:
    final: list[ContentChunk] = []
    for chunk in chunks:
        if any(SequenceMatcher(None, chunk.text, e.text).ratio() >= 0.92 for e in final):
            continue
        final.append(chunk)
    return final


@dataclass
class RetrievedContext:
    question_key: str
    question: str
    chunks: list[ContentChunk]


def retrieve_contexts(store: VectorStore, top_k: int, max_final_chunks: int) -> list[RetrievedContext]:
    outputs: list[RetrievedContext] = []
    for key, question in QUESTION_SET.items():
        ids, scores = _search(store, question, top_k=top_k)
        scored = [
            (store.chunks[idx], _boost_score(store.chunks[idx], float(sc), key))
            for idx, sc in zip(ids, scores, strict=False)
            if 0 <= idx < len(store.chunks)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        picked = _dedupe([c for c, _ in scored])[:max_final_chunks]
        outputs.append(RetrievedContext(question_key=key, question=question, chunks=picked))
    return outputs
