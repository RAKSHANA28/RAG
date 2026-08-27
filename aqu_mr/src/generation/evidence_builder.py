"""
Evidence Builder for AQU-MR
Assembles structured, page-separated, modality-isolated evidence packets.
Never flattens into an unstructured wall of text.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..utils.schemas import (
    EvidenceItem,
    EvidencePacket,
    QueryContract,
    ModalityType,
)


class EvidenceBuilder:
    """Builds clean, structured evidence packets maintaining page and modality boundaries."""

    def __init__(self, max_total_chars: int = 4000):
        self.max_total_chars = max_total_chars

    def build_packet(
        self,
        query: str,
        query_contract: QueryContract,
        evidence_items: List[EvidenceItem]
    ) -> EvidencePacket:
        """
        Structures evidence items by page and modality into a coherent, provenance-tracked packet.
        """
        # Group items by page number
        items_by_page: Dict[int, List[EvidenceItem]] = defaultdict(list)
        for item in evidence_items:
            items_by_page[item.page_num].append(item)

        sorted_pages = sorted(items_by_page.keys())

        # Construct structured evidence text
        packet_lines = [
            f"=== MULTIMODAL EVIDENCE PACKET ===",
            f"Target Query: {query}",
            f"Query Intent: {query_contract.intent.value}",
            f"Modality Prior: Text={query_contract.modality_weights.get('text', 0.4):.2f}, "
            f"Table={query_contract.modality_weights.get('table', 0.4):.2f}, "
            f"Image={query_contract.modality_weights.get('image', 0.2):.2f}",
            f"Expected Answer Type: {query_contract.expected_answer_type}",
            ""
        ]

        total_chars = sum(len(l) for l in packet_lines)

        for p_num in sorted_pages:
            p_items = items_by_page[p_num]
            packet_lines.append(f"--- [PAGE {p_num} EVIDENCE] ---")
            
            # 1. Page Tables
            table_items = [it for it in p_items if it.modality == ModalityType.TABLE]
            if table_items:
                packet_lines.append(f"  [MODALITY: TABLES (Page {p_num})]")
                for t in table_items:
                    table_block = f"  Table ID: {t.evidence_id}\n{t.content}"
                    packet_lines.append(table_block)
                    packet_lines.append("")

            # 2. Page Text Chunks
            text_items = [it for it in p_items if it.modality == ModalityType.TEXT]
            if text_items:
                packet_lines.append(f"  [MODALITY: TEXT (Page {p_num})]")
                for txt in text_items:
                    text_block = f"  Chunk ({txt.evidence_id}): {txt.content}"
                    packet_lines.append(text_block)
                    packet_lines.append("")

            # 3. Page Visuals / Images
            image_items = [it for it in p_items if it.modality == ModalityType.IMAGE]
            if image_items:
                packet_lines.append(f"  [MODALITY: VISUAL / IMAGE (Page {p_num})]")
                for img in image_items:
                    img_block = f"  Visual Item ({img.evidence_id}): File={img.content} | OCR Context={img.provenance_details.get('ocr_snippet', '')}"
                    packet_lines.append(img_block)
                    packet_lines.append("")

            packet_lines.append(f"--- [END PAGE {p_num}] ---\n")

        raw_evidence_text = "\n".join(packet_lines)

        return EvidencePacket(
            query=query,
            query_contract=query_contract,
            pages_referenced=sorted_pages,
            items_by_page=dict(items_by_page),
            raw_evidence_text=raw_evidence_text,
            total_items=len(evidence_items)
        )
