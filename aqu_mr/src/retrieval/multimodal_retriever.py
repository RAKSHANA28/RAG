"""
Multimodal Retrieval and Page-Level Fusion Engine
Combines text, table, and image evidence using adaptive modality weights and soft page fusion.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from ..utils.schemas import (
    Document,
    QueryContract,
    EvidenceItem,
    EvidencePacket,
    ModalityType,
)
from .query_analyzer import QueryAnalyzer
from .text_retriever import TextRetriever
from .table_retriever import TableRetriever
from .image_retriever import ImageRetriever


class MultimodalRetriever:
    """Orchestrates modality-aware retrieval with soft page-level fusion."""

    def __init__(
        self,
        top_k_pages: int = 3,
        top_k_chunks: int = 5,
        top_k_tables: int = 3,
        top_k_images: int = 3,
        evidence_budget_chars: int = 4000
    ):
        self.top_k_pages = top_k_pages
        self.top_k_chunks = top_k_chunks
        self.top_k_tables = top_k_tables
        self.top_k_images = top_k_images
        self.evidence_budget_chars = evidence_budget_chars

        self.query_analyzer = QueryAnalyzer()
        self.text_retriever = TextRetriever()
        self.table_retriever = TableRetriever()
        self.image_retriever = ImageRetriever()

        self.current_document: Optional[Document] = None

    def index_document(self, document: Document):
        """Indexes all document modalities."""
        self.current_document = document
        self.text_retriever.index_document(document)
        self.table_retriever.index_document(document)
        self.image_retriever.index_document(document)

    def retrieve(
        self,
        query: str,
        query_contract: Optional[QueryContract] = None
    ) -> Tuple[List[EvidenceItem], List[int], Dict[str, Any]]:
        """
        Executes soft modality-aware retrieval and page-level fusion.
        Returns:
            (selected_evidence_items, ranked_page_numbers, fusion_debug_info)
        """
        if self.current_document is None:
            raise ValueError("No document indexed. Call index_document() first.")

        # 1. Query Analysis if not provided
        if query_contract is None:
            query_contract = self.query_analyzer.analyze(query)

        mod_weights = query_contract.modality_weights
        w_text = mod_weights.get("text", 0.4)
        w_table = mod_weights.get("table", 0.4)
        w_image = mod_weights.get("image", 0.2)
        entities = query_contract.key_entities

        # 2. Retrieve candidates across modalities
        text_items = self.text_retriever.retrieve(
            query=query,
            top_k=self.top_k_chunks,
            target_entities=entities
        )
        table_items = self.table_retriever.retrieve(
            query=query,
            top_k=self.top_k_tables,
            target_entities=entities
        )
        image_items = self.image_retriever.retrieve(
            query=query,
            top_k=self.top_k_images,
            target_entities=entities
        )

        # 3. Soft Page-Level Fusion
        page_modality_scores: Dict[int, Dict[str, float]] = defaultdict(lambda: {"text": 0.0, "table": 0.0, "image": 0.0})
        page_items: Dict[int, List[EvidenceItem]] = defaultdict(list)

        for item in text_items:
            page_modality_scores[item.page_num]["text"] = max(page_modality_scores[item.page_num]["text"], item.score)
            page_items[item.page_num].append(item)

        for item in table_items:
            page_modality_scores[item.page_num]["table"] = max(page_modality_scores[item.page_num]["table"], item.score)
            page_items[item.page_num].append(item)

        for item in image_items:
            page_modality_scores[item.page_num]["image"] = max(page_modality_scores[item.page_num]["image"], item.score)
            page_items[item.page_num].append(item)

        # If a page had no hits, ensure all existing pages are represented with minimal baseline
        for page in self.current_document.pages:
            if page.page_num not in page_modality_scores:
                page_modality_scores[page.page_num] = {"text": 0.01, "table": 0.01, "image": 0.01}

        # Calculate composite page scores
        fused_page_scores: Dict[int, float] = {}
        for p_num, scores in page_modality_scores.items():
            fused = (
                w_text * scores["text"] +
                w_table * scores["table"] +
                w_image * scores["image"]
            )
            fused_page_scores[p_num] = round(float(fused), 4)

        # Rank pages by composite score
        ranked_pages = sorted(fused_page_scores.keys(), key=lambda p: fused_page_scores[p], reverse=True)
        top_pages = ranked_pages[:self.top_k_pages]

        # 4. Collect evidence within evidence budget
        selected_evidence: List[EvidenceItem] = []
        current_char_count = 0

        # Prioritize evidence from top pages
        for p_num in top_pages:
            items_for_p = page_items.get(p_num, [])
            # Sort items by item score * modality weight
            def item_priority(item: EvidenceItem):
                w = w_text if item.modality == ModalityType.TEXT else (w_table if item.modality == ModalityType.TABLE else w_image)
                return item.score * w

            items_for_p.sort(key=item_priority, reverse=True)
            for item in items_for_p:
                item_len = len(item.content)
                if current_char_count + item_len <= self.evidence_budget_chars or len(selected_evidence) < 2:
                    selected_evidence.append(item)
                    current_char_count += item_len

        # If still empty (e.g. initial cold search), include page 1 thumbnail/text
        if not selected_evidence and self.current_document.pages:
            first_p = self.current_document.pages[0]
            if first_p.text_chunks:
                selected_evidence.append(EvidenceItem(
                    evidence_id="fallback_p1_text",
                    doc_id=first_p.doc_id,
                    page_num=first_p.page_num,
                    modality=ModalityType.TEXT,
                    content=first_p.text_chunks[0].text,
                    formatted_content=f"**[Page 1 Text]**\n{first_p.text_chunks[0].text}",
                    score=0.1,
                    provenance_details={"page": 1, "fallback": True}
                ))

        debug_info = {
            "query_contract": query_contract.raw_analysis,
            "modality_weights": mod_weights,
            "raw_text_hits": len(text_items),
            "raw_table_hits": len(table_items),
            "raw_image_hits": len(image_items),
            "page_fusion_scores": {f"Page {p}": score for p, score in fused_page_scores.items()},
            "ranked_pages": ranked_pages,
            "selected_page_count": len(top_pages),
            "selected_evidence_count": len(selected_evidence)
        }

        return selected_evidence, top_pages, debug_info
