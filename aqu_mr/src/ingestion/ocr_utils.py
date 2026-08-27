"""
OCR and Layout Analysis Utilities
Extracts OCR tokens, bounding boxes, and hierarchical text blocks.
"""

from typing import List, Dict, Any, Optional, Tuple
import os
import re
from PIL import Image
from ..utils.schemas import BoundingBox


class OCREngine:
    """Multi-engine OCR processor with graceful fallbacks."""

    def __init__(self, engine_type: str = "auto"):
        self.engine_type = engine_type
        self._tesseract_available = False
        self._check_tesseract()

    def _check_tesseract(self):
        try:
            import pytesseract
            # Test if tesseract binary is actually accessible
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception:
            self._tesseract_available = False

    def extract_from_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extracts raw text, cleaned text, and bounding-box words from an image.
        Returns:
            {
                "raw_text": str,
                "cleaned_text": str,
                "words": List[Dict[str, Any]], # {"text": str, "bbox": BoundingBox}
                "lines": List[str]
            }
        """
        if self._tesseract_available and (self.engine_type in ("auto", "tesseract")):
            try:
                import pytesseract
                # Get detailed bounding box data
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                raw_text = pytesseract.image_to_string(image)
                words = []
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    txt = data['text'][i].strip()
                    if txt:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        words.append({
                            "text": txt,
                            "bbox": BoundingBox(x0=x, y0=y, x1=x+w, y1=y+h),
                            "confidence": data['conf'][i]
                        })
                cleaned = self.clean_ocr_text(raw_text)
                return {
                    "raw_text": raw_text,
                    "cleaned_text": cleaned,
                    "words": words,
                    "lines": [line.strip() for line in raw_text.splitlines() if line.strip()]
                }
            except Exception:
                pass

        # Fallback when OCR binary is not directly available on machine
        return self._heuristic_image_ocr(image)

    def _heuristic_image_ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Conservative layout-aware fallback when external OCR binary is unavailable."""
        w, h = image.size
        # Fallback placeholder indicating page visual dimensions
        placeholder_text = f"[Image Page: {w}x{h} px]"
        return {
            "raw_text": placeholder_text,
            "cleaned_text": placeholder_text,
            "words": [],
            "lines": [placeholder_text]
        }

    @staticmethod
    def clean_ocr_text(text: str) -> str:
        """
        Conservative text cleanup:
        Preserves numbers, punctuation, dates, units, and structural layout.
        Removes only excessive repeated whitespace and obvious extraction artifacts.
        """
        if not text:
            return ""
        # Normalize carriage returns and line feeds
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove non-printable control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Collapse multiple horizontal whitespace while preserving newlines
        lines = []
        for line in text.split('\n'):
            cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
            lines.append(cleaned_line)
        # Join lines, preserving empty lines for paragraph separation (max 2 consecutive newlines)
        cleaned = '\n'.join(lines)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()
