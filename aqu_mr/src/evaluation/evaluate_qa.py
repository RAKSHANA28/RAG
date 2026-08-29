"""
End-to-End QA Evaluation Module for AQU-MR
Computes ANLS, Exact Match, F1, Hallucination/Unsupported Claim Rate, and Abstention Accuracy
with modality-level breakdowns.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from ..utils.schemas import Document
from ..ingestion.document_loader import DocumentLoader
from ..retrieval.query_analyzer import QueryAnalyzer
from ..retrieval.multimodal_retriever import MultimodalRetriever
from ..generation.evidence_builder import EvidenceBuilder
from ..generation.answer_generator import GroundedAnswerGenerator
from ..generation.answer_validator import AnswerValidator
from .metrics import calculate_anls, calculate_exact_match, calculate_f1_score
from .docvqa_loader import DocVQABenchmarkLoader


class QAEvaluator:
    """Evaluates end-to-end question answering pipeline against DocVQA benchmarks."""

    def __init__(self):
        self.doc_loader = DocumentLoader()
        self.query_analyzer = QueryAnalyzer()
        self.retriever = MultimodalRetriever()
        self.evidence_builder = EvidenceBuilder()
        self.generator = GroundedAnswerGenerator()
        self.validator = AnswerValidator()
        self.benchmark_loader = DocVQABenchmarkLoader()

    def evaluate_benchmark(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Runs full benchmark QA evaluation."""
        samples = self.benchmark_loader.load_samples()
        if sample_limit:
            samples = samples[:sample_limit]

        anls_scores = []
        em_scores = []
        f1_scores = []
        unsupported_claims_count = 0
        correct_abstentions = 0
        total_unanswerable = 0

        modality_anls: Dict[str, List[float]] = {"text": [], "table": [], "visual": [], "multimodal": []}
        predictions_log = []

        indexed_docs: Dict[str, Document] = {}

        for item in samples:
            doc_id = item["doc_id"]
            question = item["question"]
            gold_answers = item["answers"]
            modality = item.get("modality", "text")
            is_unanswerable = item.get("is_unanswerable", False)

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

            # 1. Query Analysis
            q_contract = self.query_analyzer.analyze(question)

            # 2. Retrieval & Fusion
            selected_evidence, ranked_pages, _ = self.retriever.retrieve(question, q_contract)

            # 3. Evidence Building
            evidence_packet = self.evidence_builder.build_packet(question, q_contract, selected_evidence)

            # 4. Generation
            answer = self.generator.generate(question, q_contract, evidence_packet)

            # 5. Validation
            val_result = self.validator.validate(answer, evidence_packet)
            if not val_result.is_valid and val_result.unsupported_claims:
                unsupported_claims_count += 1

            # 6. Metrics
            anls = calculate_anls(answer.answer_text, gold_answers)
            em = calculate_exact_match(answer.answer_text, gold_answers)
            f1 = calculate_f1_score(answer.answer_text, gold_answers)

            anls_scores.append(anls)
            em_scores.append(em)
            f1_scores.append(f1)

            if modality in modality_anls:
                modality_anls[modality].append(anls)

            if is_unanswerable:
                total_unanswerable += 1
                if answer.is_abstention:
                    correct_abstentions += 1

            predictions_log.append({
                "question_id": item.get("question_id"),
                "question": question,
                "prediction": answer.answer_text,
                "gold_answers": gold_answers,
                "modality": modality,
                "anls": round(anls, 4),
                "exact_match": em,
                "grounding_score": val_result.grounding_score,
                "is_valid": val_result.is_valid
            })

        accuracy_scores = [1.0 if s >= 0.5 else 0.0 for s in anls_scores]
        abstention_rate = (correct_abstentions / total_unanswerable) if total_unanswerable > 0 else 1.0
        hallucination_rate = (unsupported_claims_count / len(samples)) if samples else 0.0

        modality_breakdown = {}
        for m, scores in modality_anls.items():
            if scores:
                m_samples = [p for p in predictions_log if p["modality"] == m]
                m_em = [p["exact_match"] for p in m_samples]
                modality_breakdown[f"{m.capitalize()}_QA"] = {
                    "ANLS": round(float(np.mean(scores)), 4),
                    "Exact_Match": round(float(np.mean(m_em)), 4) if m_em else 0.0,
                    "Count": len(scores)
                }

        summary = {
            "evaluation_layer": "aqu_mr/src/evaluation/",
            "dataset": "VLR-CVC/DocVQA-2026",
            "total_evaluated_samples": len(samples),
            "metrics": {
                "ANLS": round(float(np.mean(anls_scores)), 4) if anls_scores else 0.0,
                "Exact_Match": round(float(np.mean(em_scores)), 4) if em_scores else 0.0,
                "Accuracy": round(float(np.mean(accuracy_scores)), 4) if accuracy_scores else 0.0,
                "F1_Score": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
                "Unsupported_Claim_Rate": round(float(hallucination_rate), 4),
                "Hallucination_Rate": round(float(hallucination_rate), 4),
                "Correct_Abstention_Rate": round(float(abstention_rate), 4)
            },
            "modality_breakdown": modality_breakdown,
            "sample_predictions": predictions_log
        }

        # Save to JSON file
        out_dir = "data/evaluation"
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "eval_matrix.json")
        try:
            import json
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        return summary
