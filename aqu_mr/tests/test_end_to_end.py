"""
End-to-End Pipeline Tests for AQU-MR
"""

import pytest
from aqu_mr.src.ingestion.document_loader import DocumentLoader
from aqu_mr.src.retrieval.query_analyzer import QueryAnalyzer
from aqu_mr.src.retrieval.multimodal_retriever import MultimodalRetriever
from aqu_mr.src.generation.evidence_builder import EvidenceBuilder
from aqu_mr.src.generation.answer_generator import GroundedAnswerGenerator
from aqu_mr.src.generation.answer_validator import AnswerValidator
from aqu_mr.src.evaluation.docvqa_loader import DocVQABenchmarkLoader
from aqu_mr.src.evaluation.metrics import calculate_anls


def test_end_to_end_docvqa_sample():
    # 1. Load benchmark sample
    loader = DocVQABenchmarkLoader()
    samples = loader.load_samples()
    assert len(samples) > 0

    sample = samples[0] # "What was the total revenue in Q3 2025?" -> "$4.82B"
    doc_loader = DocumentLoader()
    doc = doc_loader.from_docvqa_images(
        doc_id=sample["doc_id"],
        page_images=sample["page_images"],
        ocr_annotations=sample.get("ocr_annotations")
    )

    # 2. Analyze query
    analyzer = QueryAnalyzer()
    q_contract = analyzer.analyze(sample["question"])

    # 3. Retrieve
    retriever = MultimodalRetriever()
    retriever.index_document(doc)
    evidence, pages, _ = retriever.retrieve(sample["question"], q_contract)
    assert len(evidence) > 0
    assert 1 in pages

    # 4. Assemble Evidence
    builder = EvidenceBuilder()
    packet = builder.build_packet(sample["question"], q_contract, evidence)

    # 5. Generate Answer
    generator = GroundedAnswerGenerator(provider="extractive")
    answer = generator.generate(sample["question"], q_contract, packet)
    assert answer.answer_text != ""

    # 6. Validate
    validator = AnswerValidator()
    val_res = validator.validate(answer, packet)
    assert val_res.is_valid is True

    # 7. Check ANLS
    anls = calculate_anls(answer.answer_text, sample["answers"])
    assert anls > 0.5
