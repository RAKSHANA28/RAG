"""
Tests for Ingestion Layer (Document Loading, Text Processing, Table Parsing, Image Features)
"""

import os
import pytest
from PIL import Image

from aqu_mr.src.ingestion.document_loader import DocumentLoader
from aqu_mr.src.ingestion.text_processor import TextProcessor
from aqu_mr.src.ingestion.table_processor import TableProcessor
from aqu_mr.src.ingestion.image_processor import ImageProcessor
from aqu_mr.src.utils.schemas import ModalityType


def test_text_processor_chunking():
    processor = TextProcessor(chunk_size=40, chunk_overlap=10)
    raw_text = "This is paragraph 1 with some information.\n\nThis is paragraph 2 with other facts.\n\nParagraph 3."
    chunks = processor.create_chunks("doc_test", 1, raw_text)
    
    assert len(chunks) >= 2
    assert all(c.page_num == 1 for c in chunks)
    assert all(c.doc_id == "doc_test" for c in chunks)


def test_table_processor_markdown():
    processor = TableProcessor()
    md_text = """
    Here is a financial report:
    | Quarter | Revenue | Profit |
    |---|---|---|
    | Q1 | $10M | $2M |
    | Q2 | $15M | $4M |
    """
    tables = processor.extract_tables_from_text("doc_tbl", 1, md_text)
    assert len(tables) == 1
    tbl = tables[0]
    assert len(tbl.headers) == 3
    assert tbl.headers[0] == "Quarter"
    assert len(tbl.rows) == 2
    assert tbl.rows[0][1] == "$10M"
    assert tbl.cell_matrix[0][1].value == "$10M"
    assert tbl.cell_matrix[0][1].col_header == "Revenue"


def test_image_processor_features(tmp_path):
    processor = ImageProcessor()
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    feat = processor.compute_visual_embedding(img)
    assert len(feat) > 0

    item = processor.process_page_image("doc_img", 1, img, str(tmp_path), "test context")
    assert os.path.exists(item.image_path)
    assert item.page_num == 1
    assert item.ocr_context == "test context"


def test_document_loader_from_docvqa(tmp_path):
    loader = DocumentLoader(cache_dir=str(tmp_path))
    img1 = Image.new("RGB", (400, 500), color=(255, 255, 255))
    doc = loader.from_docvqa_images("test_docvqa", [img1])
    
    assert doc.doc_id == "test_docvqa"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert len(doc.pages[0].images) == 1
