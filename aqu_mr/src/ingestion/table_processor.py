"""
Table Processor for AQU-MR
Preserves first-class table structures: headers, rows, columns, and cell coordinates.
Never flattens tables to plain unstructured text.
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import csv
import io
from ..utils.schemas import TableStructure, TableCell, BoundingBox


class TableProcessor:
    """Detects, extracts, and standardizes structured tables from text and document pages."""

    def __init__(self, mode: str = "structured"):
        self.mode = mode

    def extract_tables_from_text(
        self,
        doc_id: str,
        page_num: int,
        page_text: str
    ) -> List[TableStructure]:
        """
        Scans page text for structured tables (Markdown, pipe-delimited, CSV, or aligned column blocks).
        """
        tables: List[TableStructure] = []
        
        # 1. Look for Markdown / Pipe-delimited tables
        md_tables = self._extract_markdown_tables(doc_id, page_num, page_text)
        tables.extend(md_tables)

        # 2. Look for whitespace/tab-aligned columnar tables if no markdown table was found
        if not tables:
            aligned_tables = self._extract_aligned_column_tables(doc_id, page_num, page_text)
            tables.extend(aligned_tables)

        return tables

    def _extract_markdown_tables(
        self,
        doc_id: str,
        page_num: int,
        text: str
    ) -> List[TableStructure]:
        """Extracts Markdown/pipe tables (| Header 1 | Header 2 |)."""
        tables = []
        lines = text.split('\n')
        i = 0
        table_idx = 0

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('|') and line.endswith('|') and '|' in line[1:-1]:
                # Potential table start
                table_lines = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('|') and next_line.endswith('|'):
                        table_lines.append(next_line)
                        j += 1
                    else:
                        break
                
                if len(table_lines) >= 2:
                    table_struct = self._parse_pipe_table_lines(doc_id, page_num, table_idx, table_lines)
                    if table_struct:
                        tables.append(table_struct)
                        table_idx += 1
                i = j
            else:
                i += 1

        return tables

    def _parse_pipe_table_lines(
        self,
        doc_id: str,
        page_num: int,
        table_idx: int,
        lines: List[str]
    ) -> Optional[TableStructure]:
        parsed_rows = []
        for line in lines:
            # Strip outer pipes and split by pipe
            content = line.strip('|')
            cells = [c.strip() for c in content.split('|')]
            # Check if this line is a separator like |---|---|
            if all(re.match(r'^:?-+:?$', c) for c in cells if c):
                continue
            parsed_rows.append(cells)

        if not parsed_rows:
            return None

        headers = parsed_rows[0]
        data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []

        return self.create_table_structure(
            doc_id=doc_id,
            page_num=page_num,
            table_idx=table_idx,
            headers=headers,
            rows=data_rows,
            title=""
        )

    def _extract_aligned_column_tables(
        self,
        doc_id: str,
        page_num: int,
        text: str
    ) -> List[TableStructure]:
        """Detects columnar blocks where columns are separated by 2+ spaces or tabs."""
        tables = []
        lines = text.split('\n')
        i = 0
        table_idx = 0

        while i < len(lines):
            line = lines[i].rstrip()
            # Check for multiple column segments
            cols = [c.strip() for c in re.split(r'\t+|\s{2,}', line) if c.strip()]
            
            if len(cols) >= 2:
                # Potential table header or first row
                candidate_rows = [cols]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip()
                    if not next_line:
                        break
                    next_cols = [c.strip() for c in re.split(r'\t+|\s{2,}', next_line) if c.strip()]
                    if len(next_cols) >= 2:
                        candidate_rows.append(next_cols)
                        j += 1
                    else:
                        break

                # If at least 3 rows with consistent column count (or numeric data)
                if len(candidate_rows) >= 3:
                    # Normalize columns
                    max_cols = max(len(r) for r in candidate_rows)
                    headers = candidate_rows[0]
                    if len(headers) < max_cols:
                        headers.extend([f"Col {k+1}" for k in range(len(headers), max_cols)])
                    
                    data_rows = candidate_rows[1:]
                    table_struct = self.create_table_structure(
                        doc_id=doc_id,
                        page_num=page_num,
                        table_idx=table_idx,
                        headers=headers,
                        rows=data_rows,
                        title="Detected Tabular Section"
                    )
                    tables.append(table_struct)
                    table_idx += 1
                    i = j
                else:
                    i += 1
            else:
                i += 1

        return tables

    def create_table_structure(
        self,
        doc_id: str,
        page_num: int,
        table_idx: int,
        headers: List[str],
        rows: List[List[str]],
        title: str = "",
        bbox: Optional[BoundingBox] = None
    ) -> TableStructure:
        """Constructs a fully-featured TableStructure object with 2D cell matrix."""
        table_id = f"{doc_id}_p{page_num}_tbl{table_idx}"
        
        # Build cell matrix
        cell_matrix: List[List[TableCell]] = []
        for r_idx, row in enumerate(rows):
            row_cells: List[TableCell] = []
            row_hdr = row[0] if len(row) > 0 else f"Row {r_idx+1}"
            for c_idx, val in enumerate(row):
                col_hdr = headers[c_idx] if c_idx < len(headers) else f"Col {c_idx+1}"
                row_cells.append(TableCell(
                    row_idx=r_idx,
                    col_idx=c_idx,
                    value=str(val),
                    row_header=row_hdr,
                    col_header=col_hdr
                ))
            cell_matrix.append(row_cells)

        # Build CSV string
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        if headers:
            writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
        raw_csv = csv_buffer.getvalue()

        # Build Markdown string
        md_lines = []
        if title:
            md_lines.append(f"**Table: {title}**")
        if headers:
            md_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            padded = list(row) + [""] * max(0, len(headers) - len(row))
            md_lines.append("| " + " | ".join(str(c) for c in padded[:len(headers)]) + " |")
        raw_markdown = "\n".join(md_lines)

        return TableStructure(
            table_id=table_id,
            doc_id=doc_id,
            page_num=page_num,
            title=title,
            headers=headers,
            rows=rows,
            raw_csv=raw_csv,
            raw_markdown=raw_markdown,
            cell_matrix=cell_matrix,
            bbox=bbox,
            metadata={"row_count": len(rows), "col_count": len(headers)}
        )
