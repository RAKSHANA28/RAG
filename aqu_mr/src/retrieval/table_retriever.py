"""
Structured Table Retriever for AQU-MR
Retrieves tables via lexical header/cell matching and dense table representations.
Preserves rows, columns, headers, and exact cell coordinates.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.schemas import TableStructure, TableCell, EvidenceItem, ModalityType, Document


class TableRetriever:
    """Structured table retriever preserving 2D cell relationships."""

    def __init__(self):
        self.tables: List[TableStructure] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.table_matrix = None

    def index_document(self, document: Document):
        """Indexes all structured tables in the document."""
        self.tables = []
        for page in document.pages:
            self.tables.extend(page.tables)

        if not self.tables:
            self.vectorizer = None
            self.table_matrix = None
            return

        # Prepare corpus of serialized tables (combining title, headers, and cell values)
        corpus = []
        for t in self.tables:
            header_str = " ".join(t.headers)
            cell_strs = []
            for row in t.rows:
                cell_strs.append(" ".join(str(c) for c in row))
            corpus.append(f"{t.title} {header_str} " + " ".join(cell_strs))

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.table_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        target_entities: Optional[List[str]] = None
    ) -> List[EvidenceItem]:
        """
        Retrieves top-k relevant tables, pinpointing relevant candidate rows and cells.
        """
        if not self.tables or self.vectorizer is None or self.table_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        dense_scores = cosine_similarity(query_vec, self.table_matrix).flatten()

        query_tokens = set(re.findall(r'\b\w+\b', query.lower())) - {
            "what", "is", "the", "in", "of", "for", "to", "a", "an", "and", "or", "on", "at",
            "by", "with", "from", "about", "as", "into", "like", "tell", "me", "can", "you",
            "give", "show", "find", "does", "did", "was", "were", "are", "be", "this", "that"
        }

        results: List[EvidenceItem] = []
        scored_tables = []

        for idx, table in enumerate(self.tables):
            dense_score = float(dense_scores[idx])

            # 1. Header relevance
            header_text = " ".join(table.headers).lower()
            header_overlap = sum(1 for tok in query_tokens if tok in header_text)

            # 2. Cell value matches & target cell localization
            matched_cells = []
            row_scores = [0] * len(table.rows)

            for r_idx, row in enumerate(table.rows):
                for c_idx, cell_val in enumerate(row):
                    val_str = str(cell_val).strip()
                    val_lower = val_str.lower()
                    if not val_lower:
                        continue

                    # Check entity/token hit
                    hit = False
                    if target_entities:
                        for ent in target_entities:
                            if ent.lower() in val_lower and len(ent) > 2:
                                hit = True
                                break
                    if any(tok in val_lower for tok in query_tokens if len(tok) > 2):
                        hit = True

                    if hit:
                        row_scores[r_idx] += 1
                        col_h = table.headers[c_idx] if c_idx < len(table.headers) else f"Col {c_idx+1}"
                        row_h = row[0] if len(row) > 0 else f"Row {r_idx+1}"
                        matched_cells.append({
                            "row_idx": r_idx,
                            "col_idx": c_idx,
                            "value": val_str,
                            "col_header": col_h,
                            "row_header": row_h
                        })

            # 3. Explicit Table Number Match (e.g. "Table 1", "Table 3")
            explicit_boost = 0.0
            t_match = re.search(r'\btable\s+(\d+)\b', query.lower())
            if t_match:
                target_t_num = int(t_match.group(1))
                if (table.title and f"table {target_t_num}" in table.title.lower()) or f"tbl{target_t_num - 1}" in table.table_id:
                    explicit_boost = 5.0

            total_score = (dense_score * 0.4) + (header_overlap * 0.25) + (max(row_scores, default=0) * 0.15) + explicit_boost
            scored_tables.append((total_score, table, matched_cells))

        # Sort by composite score
        scored_tables.sort(key=lambda x: x[0], reverse=True)

        for score, table, matched_cells in scored_tables[:top_k]:
            if score <= 0.05 and not matched_cells:
                continue

            md_content = table.to_markdown()
            formatted = f"**[Page {table.page_num} Table Structure ({table.table_id})]**\n{md_content}"

            results.append(EvidenceItem(
                evidence_id=f"table_ev_{table.table_id}",
                doc_id=table.doc_id,
                page_num=table.page_num,
                modality=ModalityType.TABLE,
                content=md_content,
                formatted_content=formatted,
                score=round(float(score), 4),
                provenance_details={
                    "table_id": table.table_id,
                    "page": table.page_num,
                    "headers": table.headers,
                    "rows": table.rows,
                    "matched_cells": matched_cells[:10],
                    "total_rows": len(table.rows),
                    "total_cols": len(table.headers)
                },
                bbox=table.bbox,
                metadata=table.metadata
            ))

        return results
