"""
Text Processing and Semantic Chunking for AQU-MR
Performs conservative text normalization and chunking with strict page-level provenance.
"""

from typing import List, Dict, Any, Optional
import re
from ..utils.schemas import TextChunk, BoundingBox
from .ocr_utils import OCREngine


class TextProcessor:
    """Processes document text and produces meaningful, provenance-preserving chunks."""

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, raw_text: str) -> str:
        """Conservative text cleaning preserving entities, symbols, and formatting."""
        return OCREngine.clean_ocr_text(raw_text)

    def create_chunks(
        self,
        doc_id: str,
        page_num: int,
        raw_text: str,
        clean_text: Optional[str] = None,
        words_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[TextChunk]:
        """
        Creates semantic chunks from page text, preserving page provenance and offsets.
        Chunks are formed by respecting paragraph boundaries, sentences, and token spans.
        """
        if clean_text is None:
            clean_text = self.clean_text(raw_text)

        if not clean_text:
            return []

        # Split text into paragraphs first
        paragraphs = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [clean_text]

        chunks: List[TextChunk] = []
        chunk_idx = 0

        current_chunk_paragraphs: List[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk size, split by sentences/lines
            if para_len > self.chunk_size:
                # Flush existing buffer
                if current_chunk_paragraphs:
                    chunk_str = "\n\n".join(current_chunk_paragraphs)
                    chunks.append(self._build_chunk(doc_id, page_num, chunk_idx, chunk_str, raw_text))
                    chunk_idx += 1
                    current_chunk_paragraphs = []
                    current_len = 0

                # Split long paragraph
                sub_parts = re.split(r'(?<=[.!?])\s+', para)
                sub_buf: List[str] = []
                sub_len = 0
                for part in sub_parts:
                    if sub_len + len(part) > self.chunk_size and sub_buf:
                        chunk_str = " ".join(sub_buf)
                        chunks.append(self._build_chunk(doc_id, page_num, chunk_idx, chunk_str, raw_text))
                        chunk_idx += 1
                        # Retain overlap if possible
                        if self.chunk_overlap > 0 and len(sub_buf) > 1:
                            sub_buf = sub_buf[-1:]
                            sub_len = len(sub_buf[0])
                        else:
                            sub_buf = []
                            sub_len = 0
                    sub_buf.append(part)
                    sub_len += len(part)

                if sub_buf:
                    chunk_str = " ".join(sub_buf)
                    chunks.append(self._build_chunk(doc_id, page_num, chunk_idx, chunk_str, raw_text))
                    chunk_idx += 1
                continue

            # Check if adding this paragraph exceeds chunk size
            if current_len + para_len > self.chunk_size and current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs)
                chunks.append(self._build_chunk(doc_id, page_num, chunk_idx, chunk_str, raw_text))
                chunk_idx += 1
                current_chunk_paragraphs = [para]
                current_len = para_len
            else:
                current_chunk_paragraphs.append(para)
                current_len += para_len + 2

        # Flush remainder
        if current_chunk_paragraphs:
            chunk_str = "\n\n".join(current_chunk_paragraphs)
            chunks.append(self._build_chunk(doc_id, page_num, chunk_idx, chunk_str, raw_text))

        return chunks

    def _build_chunk(
        self,
        doc_id: str,
        page_num: int,
        chunk_idx: int,
        clean_chunk_text: str,
        full_raw_text: str
    ) -> TextChunk:
        chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
        
        # Calculate character offsets if present
        char_start = full_raw_text.find(clean_chunk_text[:30]) if clean_chunk_text else 0
        if char_start == -1:
            char_start = 0
        char_end = char_start + len(clean_chunk_text)

        return TextChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            page_num=page_num,
            text=clean_chunk_text,
            raw_text=clean_chunk_text,
            char_start=char_start,
            char_end=char_end,
            metadata={"chunk_index": chunk_idx}
        )
