"""
Unified Document Ingestion Pipeline for AQU-MR
Supports:
1. `from_docvqa_images`: Image-first DocVQA benchmark loader.
2. `from_pdf`: Unseen user-uploaded PDF documents.
3. `from_image`: Unseen single/multi-page image documents.
4. `from_docx`: Word document specifications.

Both paths yield identical downstream representations retaining full modality and page provenance.
"""

from typing import List, Dict, Any, Optional, Union
import os
import io
from PIL import Image, ImageDraw, ImageFont
import pypdf
import docx

from ..utils.schemas import (
    Document,
    DocumentPage,
    TextChunk,
    TableStructure,
    ImageItem,
    BoundingBox,
    ModalityType,
)
from .ocr_utils import OCREngine
from .text_processor import TextProcessor
from .table_processor import TableProcessor
from .image_processor import ImageProcessor


class DocumentLoader:
    """Unified multimodal document loader."""

    def __init__(
        self,
        cache_dir: str = "data/processed",
        chunk_size: int = 350,
        chunk_overlap: int = 75,
        ocr_engine_type: str = "auto"
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.ocr_engine = OCREngine(engine_type=ocr_engine_type)
        self.text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()

    def from_docvqa_images(
        self,
        doc_id: str,
        page_images: List[Union[str, Image.Image]],
        ocr_annotations: Optional[List[Dict[str, Any]]] = None
    ) -> Document:
        """
        Loads DocVQA benchmark image-first documents.
        Processes OCR, extracts tabular layouts, computes visual descriptors, and generates provenance.
        """
        doc_cache_dir = os.path.join(self.cache_dir, doc_id)
        os.makedirs(doc_cache_dir, exist_ok=True)

        pages: List[DocumentPage] = []

        for idx, img_input in enumerate(page_images):
            page_num = idx + 1
            if isinstance(img_input, str):
                image = Image.open(img_input).convert("RGB")
            else:
                image = img_input.convert("RGB")

            # 1. OCR Extraction (or use provided official DocVQA OCR annotations if available)
            if ocr_annotations and idx < len(ocr_annotations) and ocr_annotations[idx]:
                ocr_data = ocr_annotations[idx]
                raw_text = ocr_data.get("raw_text", "")
                words = ocr_data.get("words", [])
            else:
                ocr_res = self.ocr_engine.extract_from_image(image)
                raw_text = ocr_res["raw_text"]
                words = ocr_res["words"]

            cleaned_text = self.text_processor.clean_text(raw_text)

            # 2. Text Chunks
            text_chunks = self.text_processor.create_chunks(
                doc_id=doc_id,
                page_num=page_num,
                raw_text=raw_text,
                clean_text=cleaned_text,
                words_metadata=words
            )

            # 3. Table Processing
            tables = self.table_processor.extract_tables_from_text(
                doc_id=doc_id,
                page_num=page_num,
                page_text=cleaned_text
            )

            # 4. Image Processing & Provenance
            image_item = self.image_processor.process_page_image(
                doc_id=doc_id,
                page_num=page_num,
                image=image,
                save_dir=doc_cache_dir,
                ocr_context=cleaned_text
            )

            w, h = image.size
            page = DocumentPage(
                doc_id=doc_id,
                page_num=page_num,
                text_chunks=text_chunks,
                tables=tables,
                images=[image_item],
                full_raw_text=raw_text,
                full_clean_text=cleaned_text,
                image_path=image_item.image_path,
                width=w,
                height=h,
                metadata={"source_type": "docvqa_image"}
            )
            pages.append(page)

        return Document(
            doc_id=doc_id,
            file_path=doc_cache_dir,
            file_name=f"{doc_id}.docvqa",
            total_pages=len(pages),
            pages=pages,
            metadata={"source": "docvqa_benchmark"}
        )

    def from_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> Document:
        """
        Loads unseen user-uploaded PDF documents.
        Extracts native text, embedded tables/images, and synthesizes page visual representations.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        file_name = os.path.basename(pdf_path)
        if not doc_id:
            doc_id = os.path.splitext(file_name)[0].replace(" ", "_").lower()

        doc_cache_dir = os.path.join(self.cache_dir, doc_id)
        os.makedirs(doc_cache_dir, exist_ok=True)

        reader = pypdf.PdfReader(pdf_path)
        pages: List[DocumentPage] = []

        for idx, pdf_page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = pdf_page.extract_text() or ""
            cleaned_text = self.text_processor.clean_text(raw_text)

            # Extract or synthesize page image representation
            img_item = self._extract_or_render_pdf_page_image(
                pdf_page=pdf_page,
                doc_id=doc_id,
                page_num=page_num,
                save_dir=doc_cache_dir,
                page_text=cleaned_text
            )

            # Text Chunks
            text_chunks = self.text_processor.create_chunks(
                doc_id=doc_id,
                page_num=page_num,
                raw_text=raw_text,
                clean_text=cleaned_text
            )

            # Tables
            tables = self.table_processor.extract_tables_from_text(
                doc_id=doc_id,
                page_num=page_num,
                page_text=cleaned_text
            )

            page = DocumentPage(
                doc_id=doc_id,
                page_num=page_num,
                text_chunks=text_chunks,
                tables=tables,
                images=[img_item] if img_item else [],
                full_raw_text=raw_text,
                full_clean_text=cleaned_text,
                image_path=img_item.image_path if img_item else None,
                width=img_item.metadata.get("width", 800) if img_item else 800,
                height=img_item.metadata.get("height", 1000) if img_item else 1000,
                metadata={"source_type": "pdf"}
            )
            pages.append(page)

        return Document(
            doc_id=doc_id,
            file_path=pdf_path,
            file_name=file_name,
            total_pages=len(pages),
            pages=pages,
            metadata={"source": "pdf_upload"}
        )

    def from_image(self, image_path: str, doc_id: Optional[str] = None) -> Document:
        """Loads a single image document (PNG, JPG, TIFF)."""
        file_name = os.path.basename(image_path)
        if not doc_id:
            doc_id = os.path.splitext(file_name)[0].replace(" ", "_").lower()
        image = Image.open(image_path)
        return self.from_docvqa_images(doc_id, [image])

    def from_docx(self, docx_path: str, doc_id: Optional[str] = None) -> Document:
        """Loads a Word document, preserving paragraphs, headings, and structured tables."""
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"DOCX not found at {docx_path}")

        file_name = os.path.basename(docx_path)
        if not doc_id:
            doc_id = os.path.splitext(file_name)[0].replace(" ", "_").lower()

        doc_cache_dir = os.path.join(self.cache_dir, doc_id)
        os.makedirs(doc_cache_dir, exist_ok=True)

        doc = docx.Document(docx_path)
        
        # Extract text paragraphs
        para_texts = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(para_texts)
        cleaned_text = self.text_processor.clean_text(full_text)

        # Extract docx tables natively as TableStructures!
        tables: List[TableStructure] = []
        for t_idx, tbl in enumerate(doc.tables):
            grid = []
            for row in tbl.rows:
                grid.append([cell.text.strip() for cell in row.cells])
            if grid:
                headers = grid[0]
                data_rows = grid[1:] if len(grid) > 1 else []
                t_struct = self.table_processor.create_table_structure(
                    doc_id=doc_id,
                    page_num=1,
                    table_idx=t_idx,
                    headers=headers,
                    rows=data_rows,
                    title=f"Table {t_idx+1}"
                )
                tables.append(t_struct)

        # Render visual page representation
        img = self._render_text_to_image(cleaned_text[:1200], title=f"Doc: {file_name}")
        img_item = self.image_processor.process_page_image(
            doc_id=doc_id,
            page_num=1,
            image=img,
            save_dir=doc_cache_dir,
            ocr_context=cleaned_text[:400]
        )

        text_chunks = self.text_processor.create_chunks(
            doc_id=doc_id,
            page_num=1,
            raw_text=full_text,
            clean_text=cleaned_text
        )

        page = DocumentPage(
            doc_id=doc_id,
            page_num=1,
            text_chunks=text_chunks,
            tables=tables,
            images=[img_item],
            full_raw_text=full_text,
            full_clean_text=cleaned_text,
            image_path=img_item.image_path,
            width=img.size[0],
            height=img.size[1],
            metadata={"source_type": "docx"}
        )

        return Document(
            doc_id=doc_id,
            file_path=docx_path,
            file_name=file_name,
            total_pages=1,
            pages=[page],
            metadata={"source": "docx_upload"}
        )

    def load_document(self, file_path_or_image: Union[str, Image.Image]) -> Document:
        """Universal loader dispatching automatically based on file extension / type."""
        if isinstance(file_path_or_image, Image.Image):
            return self.from_docvqa_images("uploaded_img", [file_path_or_image])
        
        path = str(file_path_or_image)
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self.from_pdf(path)
        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            return self.from_image(path)
        elif ext in (".docx", ".doc"):
            return self.from_docx(path)
        elif ext in (".txt", ".md", ".csv"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            img = self._render_text_to_image(content[:1200], title=os.path.basename(path))
            doc_id = os.path.splitext(os.path.basename(path))[0].lower()
            doc_cache_dir = os.path.join(self.cache_dir, doc_id)
            img_item = self.image_processor.process_page_image(doc_id, 1, img, doc_cache_dir, content[:400])
            chunks = self.text_processor.create_chunks(doc_id, 1, content)
            tables = self.table_processor.extract_tables_from_text(doc_id, 1, content)
            page = DocumentPage(
                doc_id=doc_id,
                page_num=1,
                text_chunks=chunks,
                tables=tables,
                images=[img_item],
                full_raw_text=content,
                full_clean_text=content,
                image_path=img_item.image_path,
                width=800,
                height=1000,
                metadata={"source_type": "text"}
            )
            return Document(doc_id=doc_id, file_path=path, file_name=os.path.basename(path), total_pages=1, pages=[page])
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_or_render_pdf_page_image(
        self,
        pdf_page,
        doc_id: str,
        page_num: int,
        save_dir: str,
        page_text: str
    ) -> ImageItem:
        """Extracts embedded image from PDF page or renders layout image."""
        try:
            # Check if pdf_page has embedded images
            if hasattr(pdf_page, 'images') and len(pdf_page.images) > 0:
                first_img = pdf_page.images[0]
                img = Image.open(io.BytesIO(first_img.data)).convert("RGB")
                return self.image_processor.process_page_image(
                    doc_id=doc_id,
                    page_num=page_num,
                    image=img,
                    save_dir=save_dir,
                    ocr_context=page_text[:400]
                )
        except Exception:
            pass

        # Synthesize visual layout representation
        img = self._render_text_to_image(page_text, title=f"Page {page_num}")
        return self.image_processor.process_page_image(
            doc_id=doc_id,
            page_num=page_num,
            image=img,
            save_dir=save_dir,
            ocr_context=page_text[:400]
        )

    def _render_text_to_image(self, text: str, title: str = "Document Page", width: int = 800, height: int = 1050) -> Image.Image:
        """Creates a clean visual document page rendering with typography and margins."""
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw page border & header banner
        draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(220, 225, 230), width=2)
        draw.rectangle([(20, 20), (width - 20, 70)], fill=(245, 247, 250))
        draw.text((40, 35), title, fill=(30, 41, 59))

        # Draw text lines
        lines = text.splitlines() if text else ["[Blank Page]"]
        y = 90
        for line in lines[:55]:
            if y > height - 60:
                break
            # Truncate line length to fit
            disp_line = line[:85]
            if disp_line.startswith("#") or disp_line.startswith("Table:"):
                draw.text((40, y), disp_line, fill=(15, 23, 42))
            elif disp_line.startswith("|"):
                draw.text((40, y), disp_line, fill=(37, 99, 235))
            else:
                draw.text((40, y), disp_line, fill=(71, 85, 105))
            y += 16

        return img
