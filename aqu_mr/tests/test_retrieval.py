"""
Tests for Retrieval Layer (Query Analysis, Hybrid Text, Table, Dual Image, Modality Fusion)
"""

import pytest
from aqu_mr.src.retrieval.query_analyzer import QueryAnalyzer
from aqu_mr.src.retrieval.multimodal_retriever import MultimodalRetriever
from aqu_mr.src.ingestion.document_loader import DocumentLoader
from aqu_mr.src.utils.schemas import QueryIntent, ModalityType
from PIL import Image


def test_query_analyzer():
    analyzer = QueryAnalyzer()
    
    # Table lookup query
    q1 = "What was the total revenue in Q3 from the table?"
    contract1 = analyzer.analyze(q1)
    assert contract1.intent in (QueryIntent.TABLE_LOOKUP, QueryIntent.FACT)
    assert contract1.modality_weights["table"] > contract1.modality_weights["image"]
    assert "revenue" in q1.lower()

    # Visual layout query
    q2 = "What color is the logo and stamp in the diagram?"
    contract2 = analyzer.analyze(q2)
    assert contract2.intent == QueryIntent.VISUAL
    assert contract2.modality_weights["image"] > contract2.modality_weights["text"]

    # Comparative query
    q3 = "Compare the difference between 2024 and 2025 sales"
    contract3 = analyzer.analyze(q3)
    assert contract3.intent == QueryIntent.COMPARISON
    assert contract3.requires_comparison is True


def test_multimodal_retriever(tmp_path):
    loader = DocumentLoader(cache_dir=str(tmp_path))
    
    # Create test document with text and table
    img1 = Image.new("RGB", (300, 300), color=(255, 255, 255))
    doc = loader.from_docvqa_images("test_ret_doc", [img1])
    
    # Add a mock table and text to page 1
    doc.pages[0].tables = [
        loader.table_processor.create_table_structure(
            doc_id="test_ret_doc",
            page_num=1,
            table_idx=0,
            headers=["Department", "Budget"],
            rows=[["Engineering", "$5M"], ["Marketing", "$2M"]]
        )
    ]
    doc.pages[0].text_chunks = loader.text_processor.create_chunks(
        "test_ret_doc", 1, "The annual report highlights engineering accomplishments."
    )

    retriever = MultimodalRetriever()
    retriever.index_document(doc)

    evidence, pages, debug = retriever.retrieve("What is the engineering budget in the table?")
    assert len(pages) > 0
    assert pages[0] == 1
    assert len(evidence) > 0
    assert any(e.modality == ModalityType.TABLE for e in evidence)
