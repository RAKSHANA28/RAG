"""
Tests for Generation and Grounding Layer (Evidence Building, Grounding, Fact-checking, Abstention)
"""

import pytest
from aqu_mr.src.utils.schemas import (
    QueryContract,
    QueryIntent,
    EvidenceItem,
    EvidencePacket,
    ModalityType,
    GroundedAnswer,
)
from aqu_mr.src.generation.evidence_builder import EvidenceBuilder
from aqu_mr.src.generation.answer_generator import GroundedAnswerGenerator
from aqu_mr.src.generation.answer_validator import AnswerValidator


def test_evidence_builder():
    builder = EvidenceBuilder()
    q_contract = QueryContract(
        original_query="What was the revenue?",
        cleaned_query="What was the revenue?",
        intent=QueryIntent.FACT,
        modality_weights={"text": 0.5, "table": 0.3, "image": 0.2}
    )
    items = [
        EvidenceItem(
            evidence_id="ev1",
            doc_id="d1",
            page_num=1,
            modality=ModalityType.TEXT,
            content="Total revenue was $10M in 2025.",
            formatted_content="Total revenue was $10M in 2025.",
            score=0.9
        )
    ]
    packet = builder.build_packet("What was the revenue?", q_contract, items)
    assert packet.pages_referenced == [1]
    assert 1 in packet.items_by_page
    assert "$10M" in packet.raw_evidence_text


def test_answer_generator_extractive():
    generator = GroundedAnswerGenerator(provider="extractive")
    q_contract = QueryContract(
        original_query="What was the revenue in 2025?",
        cleaned_query="What was the revenue in 2025?",
        intent=QueryIntent.FACT,
        modality_weights={"text": 0.8, "table": 0.1, "image": 0.1},
        key_entities=["revenue", "2025"],
        expected_answer_type="NUMBER"
    )
    items = [
        EvidenceItem(
            evidence_id="ev1",
            doc_id="d1",
            page_num=1,
            modality=ModalityType.TEXT,
            content="Total revenue in 2025 reached $4.82B.",
            formatted_content="Total revenue in 2025 reached $4.82B.",
            score=0.95
        )
    ]
    builder = EvidenceBuilder()
    packet = builder.build_packet("What was the revenue in 2025?", q_contract, items)
    
    answer = generator.generate("What was the revenue in 2025?", q_contract, packet)
    assert answer.source_pages == [1]
    assert "$4.82B" in answer.answer_text or "4.82B" in answer.answer_text


def test_answer_validator_hallucination_check():
    validator = AnswerValidator()
    q_contract = QueryContract(
        original_query="What was the profit?",
        cleaned_query="What was the profit?",
        intent=QueryIntent.FACT,
        modality_weights={"text": 0.5, "table": 0.5, "image": 0.0}
    )
    items = [
        EvidenceItem(
            evidence_id="ev1",
            doc_id="d1",
            page_num=1,
            modality=ModalityType.TEXT,
            content="Profit was $2M.",
            formatted_content="Profit was $2M.",
            score=0.9
        )
    ]
    builder = EvidenceBuilder()
    packet = builder.build_packet("What was the profit?", q_contract, items)

    # Valid answer
    valid_ans = GroundedAnswer(
        query="What was the profit?",
        answer_text="$2M",
        confidence=0.9,
        source_pages=[1],
        supporting_evidence=items,
        reasoning_trace="Found in text",
        query_contract=q_contract
    )
    res = validator.validate(valid_ans, packet)
    assert res.is_valid is True
    assert res.numerical_passed is True

    # Hallucinated answer with unsupported number
    hallucinated_ans = GroundedAnswer(
        query="What was the profit?",
        answer_text="$999M",
        confidence=0.9,
        source_pages=[1],
        supporting_evidence=items,
        reasoning_trace="Unsupported",
        query_contract=q_contract
    )
    res2 = validator.validate(hallucinated_ans, packet)
    assert res2.is_valid is False
    assert res2.numerical_passed is False
    assert len(res2.unsupported_claims) > 0
