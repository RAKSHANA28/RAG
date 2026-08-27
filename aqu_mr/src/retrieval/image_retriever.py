"""
Dual-Signal Image Retriever for AQU-MR
Combines visual layout/feature similarity AND OCR page-text context matching.
Never relies solely on CLIP or visual signals alone.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.schemas import ImageItem, EvidenceItem, ModalityType, Document


class ImageRetriever:
    """Dual-signal image retriever combining visual embeddings with OCR/text context."""

    def __init__(self, visual_weight: float = 0.4, ocr_weight: float = 0.6):
        self.visual_weight = visual_weight
        self.ocr_weight = ocr_weight
        self.images: List[ImageItem] = []
        self.ocr_vectorizer: Optional[TfidfVectorizer] = None
        self.ocr_matrix = None

    def index_document(self, document: Document):
        """Indexes all images and associated page/OCR context."""
        self.images = []
        for page in document.pages:
            self.images.extend(page.images)

        if not self.images:
            self.ocr_vectorizer = None
            self.ocr_matrix = None
            return

        ocr_contexts = [img.ocr_context if img.ocr_context else f"page {img.page_num}" for img in self.images]
        self.ocr_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.ocr_matrix = self.ocr_vectorizer.fit_transform(ocr_contexts)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        query_visual_embedding: Optional[List[float]] = None,
        target_entities: Optional[List[str]] = None
    ) -> List[EvidenceItem]:
        """
        Retrieves images using dual-signal union fusion:
        1. Signal A: Visual descriptor similarity
        2. Signal B: OCR context text similarity
        """
        if not self.images:
            return []

        # 1. OCR Context Scores (Signal B)
        ocr_scores = np.zeros(len(self.images), dtype=np.float32)
        if self.ocr_vectorizer is not None and self.ocr_matrix is not None:
            q_vec = self.ocr_vectorizer.transform([query])
            ocr_scores = cosine_similarity(q_vec, self.ocr_matrix).flatten()

        # 2. Visual Embedding Scores (Signal A)
        visual_scores = np.zeros(len(self.images), dtype=np.float32)
        if query_visual_embedding is not None:
            q_emb = np.array(query_visual_embedding, dtype=np.float32)
            for idx, img in enumerate(self.images):
                if img.visual_features:
                    img_emb = np.array(img.visual_features, dtype=np.float32)
                    dot = np.dot(q_emb, img_emb)
                    norm = np.linalg.norm(q_emb) * np.linalg.norm(img_emb)
                    visual_scores[idx] = dot / norm if norm > 0 else 0.0

        # 3. Entity boost in OCR text
        entity_boosts = np.zeros(len(self.images), dtype=np.float32)
        if target_entities:
            for idx, img in enumerate(self.images):
                ctx_lower = (img.ocr_context or "").lower()
                matches = sum(1 for e in target_entities if e.lower() in ctx_lower)
                entity_boosts[idx] = matches * 0.2

        # 4. Dual-signal Union & Fusion
        fused_scores = (
            self.visual_weight * visual_scores +
            self.ocr_weight * ocr_scores +
            entity_boosts
        )

        top_indices = np.argsort(fused_scores)[::-1][:top_k]

        results: List[EvidenceItem] = []
        for idx in top_indices:
            score = float(fused_scores[idx])
            img = self.images[idx]
            if score <= 0.0 and len(self.images) > top_k:
                continue

            caption = img.caption or f"Page {img.page_num} Visual Evidence"
            formatted = (
                f"**[Page {img.page_num} Image: {caption}]**\n"
                f"Image Path: `{img.image_path}`\n"
                f"Associated OCR Context: {img.ocr_context[:250]}..."
            )

            results.append(EvidenceItem(
                evidence_id=f"img_ev_{img.image_id}",
                doc_id=img.doc_id,
                page_num=img.page_num,
                modality=ModalityType.IMAGE,
                content=img.image_path,
                formatted_content=formatted,
                score=round(score, 4),
                provenance_details={
                    "image_id": img.image_id,
                    "page": img.page_num,
                    "image_path": img.image_path,
                    "ocr_score": float(ocr_scores[idx]),
                    "visual_score": float(visual_scores[idx]),
                    "ocr_snippet": (img.ocr_context or "")
                },
                bbox=img.bbox,
                metadata=img.metadata
            ))

        return results
