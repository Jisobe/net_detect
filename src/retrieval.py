"""Dense retrieval over the troubleshooting knowledge base (sentence-transformers + ChromaDB),
with an agentic query-reformulation step when the first retrieval is low-confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import config
from .taxonomy import CATEGORIES

COLLECTION = "netdiag_kb"


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return get_embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500)


@dataclass
class Passage:
    chunk_id: str
    title: str
    section: str
    url: str
    text: str
    score: float  # cosine similarity to the query

    def citation(self) -> str:
        return f"{self.title}, {self.section}" if self.section else self.title


class VectorStore:
    def __init__(self, path=config.VECTOR_DB_DIR):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(path))

    def rebuild(self, chunks) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:  # collection did not exist yet
            pass
        col = self.client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        vectors = embed([c.embedding_text() for c in chunks])
        for start in range(0, len(chunks), 1000):
            batch = chunks[start:start + 1000]
            col.add(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start:start + 1000].tolist(),
                documents=[c.text for c in batch],
                metadatas=[{"title": c.title, "section": c.section, "url": c.url, "doc_id": c.doc_id}
                           for c in batch],
            )

    def query(self, text: str, k: int = config.TOP_K) -> list[Passage]:
        col = self.client.get_collection(COLLECTION)
        res = col.query(query_embeddings=embed([text]).tolist(), n_results=k,
                        include=["documents", "metadatas", "distances"])
        return [
            Passage(cid, meta["title"], meta["section"], meta["url"], doc, 1.0 - dist)
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                            res["metadatas"][0], res["distances"][0])
        ]


def build_query(category: str, evidence: list[str], symptom_summary: str | None = None) -> str:
    """Retrieval query = predicted category + its symptom evidence (+ optional free text)."""
    cat = CATEGORIES[category]
    q = f"{cat.title}. {cat.description}"
    if evidence:
        q += " Symptoms: " + "; ".join(evidence) + "."
    if symptom_summary:
        q += " " + symptom_summary
    return q + " Diagnose the root cause and remediation steps."


@dataclass
class RetrievalResult:
    passages: list[Passage]
    queries: list[str]
    reformulated: bool = False
    trace: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return max((p.score for p in self.passages), default=0.0)


class Retriever:
    def __init__(self, store: VectorStore | None = None, k: int = config.TOP_K,
                 threshold: float = config.RETRIEVAL_CONFIDENCE_THRESHOLD):
        self.store = store or VectorStore()
        self.k = k
        self.threshold = threshold

    def retrieve(self, category: str, evidence: list[str], symptom_summary: str | None = None,
                 alt_categories: list[str] | None = None, agentic: bool = True) -> RetrievalResult:
        query = build_query(category, evidence, symptom_summary)
        passages = self.store.query(query, self.k)
        result = RetrievalResult(passages, [query])
        result.trace.append(f"initial retrieval: best cosine {result.confidence:.3f}")
        if not agentic or result.confidence >= self.threshold:
            return result

        # Low confidence: expand the query with category terms, retrieve again, merge by best score.
        candidates = [f"{CATEGORIES[category].title}: {CATEGORIES[category].query_terms}"]
        for alt in alt_categories or []:
            if alt != category:
                candidates.append(f"{CATEGORIES[alt].title}: {CATEGORIES[alt].query_terms}")

        pool = {p.chunk_id: p for p in passages}
        for q in candidates:
            hits = self.store.query(q, self.k)
            result.queries.append(q)
            result.trace.append(f"reformulated query -> best cosine {max(h.score for h in hits):.3f}")
            for h in hits:
                if h.chunk_id not in pool or h.score > pool[h.chunk_id].score:
                    pool[h.chunk_id] = h
        result.passages = sorted(pool.values(), key=lambda p: -p.score)[: self.k]
        result.reformulated = True
        return result
