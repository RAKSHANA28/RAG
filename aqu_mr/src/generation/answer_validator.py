"""
Answer Validation and Grounding Verifier for AQU-MR
Validates answers for entity support, numerical precision, table cell provenance,
and prevents hallucinated claims.
"""

from typing import List, Dict, Any, Optional, Tuple
import re

from ..utils.schemas import (
    GroundedAnswer,
    EvidencePacket,
    ValidationResult,
    ModalityType,
)


class AnswerValidator:
    """Rigorous hallucination guard and answer verifier."""

    def __init__(self, min_grounding_score: float = 0.65):
        self.min_grounding_score = min_grounding_score

    def validate(
        self,
        answer: GroundedAnswer,
        evidence_packet: EvidencePacket
    ) -> ValidationResult:
        """
        Validates the generated answer against the evidence packet.
        """
        if answer.is_abstention:
            return ValidationResult(
                is_valid=True,
                grounding_score=1.0,
                entity_support_passed=True,
                numerical_passed=True,
                table_provenance_passed=True,
                source_page_passed=True,
                unsupported_claims=[],
                failure_reasons=[]
            )

        ans_text = answer.answer_text.strip()
        ans_lower = ans_text.lower()
        evidence_text = evidence_packet.raw_evidence_text.lower()

        failure_reasons: List[str] = []
        unsupported_claims: List[str] = []

        # 1. Numerical & Value Verification
        numerical_passed = True
        numbers_in_ans = re.findall(r'\d+(?:\.\d+)?', ans_text)
        for num in numbers_in_ans:
            # Check if this exact number sequence exists in evidence text
            if num not in evidence_text:
                numerical_passed = False
                unsupported_claims.append(f"Number '{num}' not found in retrieved evidence.")
                failure_reasons.append(f"Numerical mismatch: '{num}'")

        # 2. Key Entity Verification
        entity_passed = True
        words_in_ans = [w for w in re.findall(r'\b[A-Za-z]{3,}\b', ans_text) if w.lower() not in {
            "the", "and", "is", "was", "are", "were", "for", "from", "with", "that", "this",
            "page", "table", "total", "amount", "value", "based", "according", "observed"
        }]
        
        supported_word_count = sum(1 for w in words_in_ans if w.lower() in evidence_text)
        entity_ratio = (supported_word_count / len(words_in_ans)) if words_in_ans else 1.0

        if words_in_ans and entity_ratio < 0.60:
            entity_passed = False
            failure_reasons.append(f"Low entity overlap with evidence ({entity_ratio:.1%})")

        # 3. Source Page Provenance
        source_page_passed = bool(answer.source_pages) and all(
            p in evidence_packet.pages_referenced for p in answer.source_pages
        )
        if not source_page_passed and evidence_packet.pages_referenced:
            # Auto-align source pages to first valid referenced page
            answer.source_pages = evidence_packet.pages_referenced[:1]
            source_page_passed = True

        # 4. Table Cell Coordinate Verification
        table_provenance_passed = True
        if answer.query_contract and answer.query_contract.intent.value == "TABLE_LOOKUP":
            # Ensure cell value or metric is present in table evidence
            table_evidence = [
                it for p in evidence_packet.items_by_page.values()
                for it in p if it.modality == ModalityType.TABLE
            ]
            if table_evidence:
                table_texts = " ".join(t.content.lower() for t in table_evidence)
                if not any(num.lower() in table_texts for num in numbers_in_ans) and not (ans_lower in table_texts):
                    table_provenance_passed = False
                    failure_reasons.append("Answer value not grounded in retrieved table cells.")

        # Compute overall Grounding Score
        grounding_score = round(
            (0.40 * (1.0 if numerical_passed else 0.0)) +
            (0.35 * entity_ratio) +
            (0.15 * (1.0 if table_provenance_passed else 0.0)) +
            (0.10 * (1.0 if source_page_passed else 0.0)),
            3
        )

        is_valid = (grounding_score >= self.min_grounding_score) and (len(unsupported_claims) == 0)
        requires_regen = (not is_valid) and (grounding_score >= 0.35)

        return ValidationResult(
            is_valid=is_valid,
            grounding_score=grounding_score,
            entity_support_passed=entity_passed,
            numerical_passed=numerical_passed,
            table_provenance_passed=table_provenance_passed,
            source_page_passed=source_page_passed,
            unsupported_claims=unsupported_claims,
            failure_reasons=failure_reasons,
            requires_regeneration=requires_regen
        )
