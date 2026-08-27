"""
AQU-MR: Core Data Models and Type Schemas
Defines all structured contracts, evidence models, and verification schemas.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class ModalityType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class QueryIntent(str, Enum):
    FACT = "FACT"
    TABLE_LOOKUP = "TABLE_LOOKUP"
    VISUAL = "VISUAL"
    COMPARISON = "COMPARISON"
    LIST = "LIST"
    SUMMARY = "SUMMARY"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    page_num: int
    text: str
    raw_text: str = ""
    bbox: Optional[BoundingBox] = None
    char_start: int = 0
    char_end: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    row_idx: int
    col_idx: int
    value: str
    row_header: Optional[str] = None
    col_header: Optional[str] = None
    bbox: Optional[BoundingBox] = None


@dataclass
class TableStructure:
    table_id: str
    doc_id: str
    page_num: int
    title: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    raw_csv: str = ""
    raw_markdown: str = ""
    raw_html: str = ""
    cell_matrix: List[List[TableCell]] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        if self.raw_markdown:
            return self.raw_markdown
        if not self.headers and not self.rows:
            return ""
        lines = []
        if self.title:
            lines.append(f"**Table: {self.title}**")
        headers = self.headers if self.headers else [f"Col {i+1}" for i in range(len(self.rows[0]) if self.rows else 0)]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in self.rows:
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(str(c) for c in padded_row[:len(headers)]) + " |")
        return "\n".join(lines)


@dataclass
class ImageItem:
    image_id: str
    doc_id: str
    page_num: int
    image_path: str
    caption: str = ""
    visual_features: Optional[List[float]] = None
    ocr_context: str = ""
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentPage:
    doc_id: str
    page_num: int
    text_chunks: List[TextChunk] = field(default_factory=list)
    tables: List[TableStructure] = field(default_factory=list)
    images: List[ImageItem] = field(default_factory=list)
    full_raw_text: str = ""
    full_clean_text: str = ""
    image_path: Optional[str] = None
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    doc_id: str
    file_path: str
    file_name: str
    total_pages: int
    pages: List[DocumentPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryContract:
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    modality_weights: Dict[str, float]  # {"text": 0.4, "table": 0.4, "image": 0.2}
    key_entities: List[str] = field(default_factory=list)
    expected_answer_type: str = "TEXT"  # "NUMBER", "DATE", "ENTITY", "LIST", "BOOLEAN", "TEXT"
    requires_comparison: bool = False
    requires_temporal: bool = False
    raw_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    evidence_id: str
    doc_id: str
    page_num: int
    modality: ModalityType
    content: str
    formatted_content: str
    score: float
    provenance_details: Dict[str, Any] = field(default_factory=dict)
    bbox: Optional[BoundingBox] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePacket:
    query: str
    query_contract: QueryContract
    pages_referenced: List[int]
    items_by_page: Dict[int, List[EvidenceItem]]
    raw_evidence_text: str
    total_items: int = 0


@dataclass
class AnswerContract:
    expected_type: str
    required_entities: List[str] = field(default_factory=list)
    table_cell_target: Optional[Dict[str, Any]] = None
    numerical_constraints: Optional[Dict[str, Any]] = None
    source_pages_expected: List[int] = field(default_factory=list)


@dataclass
class GroundedAnswer:
    query: str
    answer_text: str
    confidence: float
    source_pages: List[int]
    supporting_evidence: List[EvidenceItem]
    reasoning_trace: str
    query_contract: QueryContract
    raw_model_output: str = ""
    is_abstention: bool = False
    validation_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    is_valid: bool
    grounding_score: float
    entity_support_passed: bool
    numerical_passed: bool
    table_provenance_passed: bool
    source_page_passed: bool
    unsupported_claims: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    requires_regeneration: bool = False
