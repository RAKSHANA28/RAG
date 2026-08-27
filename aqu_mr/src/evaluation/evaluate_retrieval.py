"""
Retrieval Evaluation Module for AQU-MR
Evaluates Recall@1, Recall@3, and MRR using independent ground-truth page annotations.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ..utils.schemas import Document
from ..ingestion.document_loader import DocumentLoader
from ..retrieval.multimodal_retriever import MultimodalRetriever
from .metrics import calculate_recall_at_k, calculate_mrr
from .docvqa_loader import DocVQABenchmarkLoader


class RetrievalEvaluator:
    """Evaluates page and modality retrieval accuracy."""

    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.retriever = MultimodalRetriever()
        self.benchmark_loader = DocVQABenchmarkLoader()

    def evaluate_benchmark(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Runs retrieval evaluation across the benchmark dataset."""
        samples = self.benchmark_loader.load_samples()
        if sample_limit:
            samples = samples[:sample_limit]

        r1_scores = []
        r3_scores = []
        mrr_scores = []
        modality_breakdown = {"text": [], "table": [], "visual": []}

        # Cache indexed documents
        indexed_docs: Dict[str, Document] = {}

        for item in samples:
            doc_id = item["doc_id"]
            gold_pages = item.get("gold_pages", [])
            modality = item.get("modality", "text")
            
            # Skip unanswerable / no gold page items for retrieval metric
            if not gold_pages:
                continue

            if doc_id not in indexed_docs:
                page_imgs = item["page_images"]
                doc = self.doc_loader.from_docvqa_images(
                    doc_id=doc_id,
                    page_images=page_imgs,
                    ocr_annotations=item.get("ocr_annotations")
                )
                indexed_docs[doc_id] = doc

            doc = indexed_docs[doc_id]
            self.retriever.index_document(doc)

            _, ranked_pages, _ = self.retriever.retrieve(item["question"])

            r1 = calculate_recall_at_k(ranked_pages, gold_pages, k=1)
            r3 = calculate_recall_at_k(ranked_pages, gold_pages, k=3)
            mrr = calculate_mrr(ranked_pages, gold_pages)

            r1_scores.append(r1)
            r3_scores.append(r3)
            mrr_scores.append(mrr)

            if modality in modality_breakdown:
                modality_breakdown[modality].append(r3)

        results = {
            "total_evaluated_samples": len(r1_scores),
            "Recall@1": round(float(np.mean(r1_scores)), 4) if r1_scores else 0.0,
            "Recall@3": round(float(np.mean(r3_scores)), 4) if r3_scores else 0.0,
            "MRR": round(float(np.mean(mrr_scores)), 4) if mrr_scores else 0.0,
            "Modality_Recall@3": {
                m: round(float(np.mean(scores)), 4) if scores else 0.0
                for m, scores in modality_breakdown.items()
            }
        }
        return results
