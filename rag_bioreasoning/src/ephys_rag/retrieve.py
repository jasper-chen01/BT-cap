from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ephys_rag.schema import RAGDocument

_EPHYS_RE = re.compile(r"ephys[_\s-]?([12])", re.I)


@dataclass
class RetrievalHit:
    document: RAGDocument
    score: float


class HybridRetriever:
    """Baseline TF-IDF over tool-backed documents. Week 2: add FAISS."""

    def __init__(self, documents: list[RAGDocument]):
        if not documents:
            raise ValueError("No documents to index")
        self.documents = documents
        corpus = [f"{doc.title}\n{doc.text}" for doc in documents]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    @staticmethod
    def parse_filters(query: str) -> dict[str, str]:
        filters: dict[str, str] = {}
        match = _EPHYS_RE.search(query)
        if match:
            filters["ephys"] = f"Ephys_{match.group(1)}"
        lowered = query.lower()
        if any(term in lowered for term in ("t cell", "t-cell", "tcell", "immune")):
            filters["compartment"] = "tcell"
        elif any(term in lowered for term in ("tumor", "glioma", "opc")):
            filters["compartment"] = "tumor"
        elif any(term in lowered for term in ("myeloid", "tam", "microglia")):
            filters["compartment"] = "myeloid"
        return filters

    def search(self, query: str, *, top_k: int = 10) -> list[RetrievalHit]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        hits = [
            RetrievalHit(document=doc, score=float(score))
            for doc, score in zip(self.documents, scores)
            if float(score) > 0
        ]
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def format_context(hits: list[RetrievalHit]) -> str:
    blocks = []
    for idx, hit in enumerate(hits, start=1):
        doc = hit.document
        blocks.append(f"[{idx}] ({doc.kind}, score={hit.score:.3f}) {doc.title}\n{doc.text}")
    return "\n\n".join(blocks)

