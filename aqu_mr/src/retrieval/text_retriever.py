"""
Hybrid Text Retriever for AQU-MR
Combines lexical BM25 and dense representation scoring with strict chunk and page provenance.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.schemas import TextChunk, EvidenceItem, ModalityType, Document


class TextRetriever:
    """Hybrid BM25 + Dense vector retriever for text chunks."""

    def __init__(self, bm25_weight: float = 0.5, dense_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.chunks: List[TextChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def index_document(self, document: Document):
        """Indexes all text chunks from all pages in the document."""
        self.chunks = []
        for page in document.pages:
            self.chunks.extend(page.text_chunks)

        if not self.chunks:
            self.bm25_index = None
            self.tfidf_matrix = None
            return

        # 1. Prepare BM25 index
        tokenized_corpus = [self._tokenize(c.text) for c in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        # 2. Prepare Dense / TF-IDF Vectorizer
        corpus_texts = [c.text for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus_texts)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        target_entities: Optional[List[str]] = None
    ) -> List[EvidenceItem]:
        """
        Retrieves top-k text chunks with hybrid BM25 + dense scoring and entity boosting.
        """
        if not self.chunks or self.bm25_index is None or self.tfidf_matrix is None:
            return []

        # 1. BM25 Scores
        query_tokens = self._tokenize(query)
        bm25_raw = np.array(self.bm25_index.get_scores(query_tokens), dtype=np.float32)
        bm25_norm = self._normalize_scores(bm25_raw)

        # 2. Dense / TF-IDF Scores
        query_vec = self.tfidf_vectorizer.transform([query])
        dense_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        dense_norm = self._normalize_scores(dense_scores)

        # 3. Entity overlap boost
        entity_boosts = np.zeros(len(self.chunks), dtype=np.float32)
        if target_entities:
            for idx, chunk in enumerate(self.chunks):
                chunk_lower = chunk.text.lower()
                matches = sum(1 for e in target_entities if e.lower() in chunk_lower)
                entity_boosts[idx] = matches * 0.15

        # 4. Hybrid Fusion
        combined_scores = (
            self.bm25_weight * bm25_norm +
            self.dense_weight * dense_norm +
            entity_boosts
        )

        # 5. Top-K Selection
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results: List[EvidenceItem] = []
        for rank, idx in enumerate(top_indices):
            chunk = self.chunks[idx]
            score = float(combined_scores[idx])
            if score <= 0.0:
                continue

            results.append(EvidenceItem(
                evidence_id=f"text_ev_{chunk.chunk_id}",
                doc_id=chunk.doc_id,
                page_num=chunk.page_num,
                modality=ModalityType.TEXT,
                content=chunk.text,
                formatted_content=f"**[Page {chunk.page_num} Text Chunk]**\n{chunk.text}",
                score=round(score, 4),
                provenance_details={
                    "page": chunk.page_num,
                    "chunk_id": chunk.chunk_id,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "bm25_score": float(bm25_raw[idx]),
                    "dense_score": float(dense_scores[idx]),
                    "rank": rank + 1
                },
                bbox=chunk.bbox,
                metadata=chunk.metadata
            ))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        s_max = np.max(scores) if len(scores) > 0 else 0
        s_min = np.min(scores) if len(scores) > 0 else 0
        if s_max > s_min:
            return (scores - s_min) / (s_max - s_min)
        elif s_max > 0:
            return scores / s_max
        return np.zeros_like(scores)
