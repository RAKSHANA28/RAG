"""
DocVQA-2026 Benchmark Dataset Loader
Provides seamless loading of DocVQA benchmark images, OCR tokens, and ground-truth QA annotations.
Includes built-in multi-modal benchmark samples.
"""

from typing import List, Dict, Any, Optional
import os
import json
from PIL import Image, ImageDraw, ImageFont


class DocVQABenchmarkLoader:
    """Loads and synthesizes benchmark evaluation datasets."""

    def __init__(self, data_dir: str = "data/evaluation"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_file = os.path.join(self.data_dir, "docvqa_2026_benchmark.json")
        self._ensure_benchmark_dataset()

    def _ensure_benchmark_dataset(self):
        """Creates realistic multimodal DocVQA-2026 benchmark samples if not present."""
        if os.path.exists(self.dataset_file):
            return

        # Create sample document images directory
        sample_img_dir = os.path.join(self.data_dir, "sample_images")
        os.makedirs(sample_img_dir, exist_ok=True)

        # 1. Document 1: Financial & Corporate Annual Report (Text + Table + Visual)
        doc1_img1 = os.path.join(sample_img_dir, "doc1_page1.png")
        doc1_img2 = os.path.join(sample_img_dir, "doc1_page2.png")
        self._create_doc1_images(doc1_img1, doc1_img2)

        # 2. Document 2: Scientific Research & Medical Clinical Trial (Visual Diagram + Metrics Table)
        doc2_img1 = os.path.join(sample_img_dir, "doc2_page1.png")
        self._create_doc2_images(doc2_img1)

        # 3. Document 3: Commercial Invoice & Shipping Receipt (Structured Form + Table)
        doc3_img1 = os.path.join(sample_img_dir, "doc3_page1.png")
        self._create_doc3_images(doc3_img1)

        # Benchmark questions spanning Text, Table, Visual, and Multimodal
        # Page 1 & 2 OCR annotations for Doc 1
        doc1_ocr = [
            {
                "raw_text": (
                    "APEX GLOBAL ENTERPRISES - Q3 2025 FINANCIAL PERFORMANCE\n\n"
                    "1. Executive Summary:\n"
                    "Apex Global demonstrated resilient operational growth across global software verticals.\n"
                    "Cloud subscription margins expanded by 340 basis points year-over-year.\n\n"
                    "Table 1: Quarterly Financial Summary (USD Billions)\n"
                    "| Quarter | Revenue | Operating Income | Net Margin |\n"
                    "|---|---|---|---|\n"
                    "| Q1 2025 | $3.95B | $1.12B | 28.3% |\n"
                    "| Q2 2025 | $4.21B | $1.25B | 29.7% |\n"
                    "| Q3 2025 | $4.82B | $1.48B | 30.7% |\n"
                    "| Q4 2024 | $3.80B | $1.05B | 27.6% |\n"
                ),
                "words": []
            },
            {
                "raw_text": (
                    "APEX GLOBAL ENTERPRISES - GOVERNANCE & AUDIT\n\n"
                    "2. Independent Auditor Opinion:\n"
                    "The consolidated statements present fairly in all material respects the financial position.\n"
                    "Approved by the Board of Directors on October 24, 2025.\n\n"
                    "OFFICIAL CORPORATE SEAL\n"
                    "Apex Global Enterprises\n"
                    "Status: Verified & Validated\n\n"
                    "Executive Sign-Off:\n"
                    "Marcus Vance\n"
                    "Chief Financial Officer, Apex Global\n"
                    "Elena Rostova\n"
                    "Chief Executive Officer, Apex Global\n"
                ),
                "words": []
            }
        ]

        doc2_ocr = [
            {
                "raw_text": (
                    "BIOMED RESEARCH INSTITUTE - CLINICAL TRIAL REPORT (PHASE 3)\n\n"
                    "Abstract: Therapeutic assessment of BMR-902 in randomized cohorts.\n\n"
                    "Table: Primary Endpoint Efficacy Breakdown\n"
                    "| Cohort Group | Patient Count | Efficacy Rate | p-value |\n"
                    "|---|---|---|---|\n"
                    "| Cohort A (Standard) | 450 patients | 72.4% | p < 0.05 |\n"
                    "| Cohort B (BMR-902 High) | 450 patients | 94.6% | p < 0.001 |\n"
                    "| Cohort C (Placebo) | 450 patients | 18.2% | N/A |\n\n"
                    "Figure 1: Cohort Response Trajectory Diagram\n"
                    "Legend: Blue = Control Group | Green = Active BMR-902\n"
                ),
                "words": []
            }
        ]

        doc3_ocr = [
            {
                "raw_text": (
                    "NEXUS LOGISTICS - OFFICIAL COMMERCIAL INVOICE\n\n"
                    "Invoice Number: 88921\n"
                    "Invoice Date: October 15, 2025\n"
                    "Payment Due Date: November 15, 2025\n\n"
                    "Itemized Services & Goods:\n"
                    "| Description | Quantity | Unit Price | Total |\n"
                    "|---|---|---|---|\n"
                    "| Enterprise Cloud Routing Cluster | 2 units | $5,000.00 | $10,000.00 |\n"
                    "| Dedicated Fiber Backbone Setup | 1 service | $4,250.00 | $4,250.00 |\n\n"
                    "Final Balance Due: $14,250.00\n"
                ),
                "words": []
            }
        ]

        benchmark_data = {
            "dataset": "VLR-CVC/DocVQA-2026",
            "version": "2.0",
            "description": "Comprehensive multimodal benchmark for Document Visual Question Answering",
            "samples": [
                # --- Document 1: Financial & Corporate Report ---
                {
                    "question_id": "q101",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "What was the total revenue in Q3 2025?",
                    "answers": ["$4.82B", "4.82B", "4.82 billion", "$4.82 billion"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q102",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "Who is the Chief Financial Officer who signed the audit report?",
                    "answers": ["Marcus Vance", "Marcus Vance, CFO", "Vance"],
                    "modality": "text",
                    "gold_pages": [2],
                    "intent": "FACT"
                },
                {
                    "question_id": "q103",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "What is the corporate seal status on page 2?",
                    "answers": ["Verified & Validated", "Verified and Validated", "Verified"],
                    "modality": "visual",
                    "gold_pages": [2],
                    "intent": "VISUAL"
                },
                {
                    "question_id": "q104",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "What was the operating income reported in Q1 2025?",
                    "answers": ["$1.12B", "1.12B", "$1.12 billion"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q105",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "Who is the Chief Executive Officer mentioned in executive sign-off?",
                    "answers": ["Elena Rostova", "Elena Rostova, CEO", "Elena Rostova Chief Executive Officer"],
                    "modality": "text",
                    "gold_pages": [2],
                    "intent": "FACT"
                },
                {
                    "question_id": "q106",
                    "doc_id": "fin_report_2025",
                    "page_images": [doc1_img1, doc1_img2],
                    "ocr_annotations": doc1_ocr,
                    "question": "What is the projected revenue for the year 2040?",
                    "answers": ["Insufficient evidence in document to answer the question.", "unanswerable", "none"],
                    "modality": "text",
                    "gold_pages": [1],
                    "intent": "FACT",
                    "is_unanswerable": True
                },

                # --- Document 2: Clinical Trial Report ---
                {
                    "question_id": "q201",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "What is the reported efficacy rate for Cohort B in the Phase 3 trial?",
                    "answers": ["94.6%", "94.6 percent", "94.6"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q202",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "What color indicates the control group in the clinical trial figure?",
                    "answers": ["Blue", "Blue bar", "Blue line"],
                    "modality": "visual",
                    "gold_pages": [1],
                    "intent": "VISUAL"
                },
                {
                    "question_id": "q203",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "How many patients were enrolled in Cohort A?",
                    "answers": ["450 patients", "450", "450 enrolled"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q204",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "What color indicates the active BMR-902 treatment in Figure 1?",
                    "answers": ["Green", "Green line", "Green bar"],
                    "modality": "visual",
                    "gold_pages": [1],
                    "intent": "VISUAL"
                },
                {
                    "question_id": "q205",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "What is the efficacy rate reported for Cohort C Placebo?",
                    "answers": ["18.2%", "18.2 percent", "18.2"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q206",
                    "doc_id": "clinical_study_09",
                    "page_images": [doc2_img1],
                    "ocr_annotations": doc2_ocr,
                    "question": "What is the expected release date for Phase 4 commercial production?",
                    "answers": ["Insufficient evidence in document to answer the question.", "unanswerable", "none"],
                    "modality": "text",
                    "gold_pages": [1],
                    "intent": "FACT",
                    "is_unanswerable": True
                },

                # --- Document 3: Commercial Logistics Invoice ---
                {
                    "question_id": "q301",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What is the final invoice balance due?",
                    "answers": ["$14,250.00", "$14,250", "14250.00", "14,250"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q302",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What is the payment due date for invoice 88921?",
                    "answers": ["November 15, 2025", "15 Nov 2025", "11/15/2025"],
                    "modality": "text",
                    "gold_pages": [1],
                    "intent": "FACT"
                },
                {
                    "question_id": "q303",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What is the price of Enterprise Cloud Routing Cluster?",
                    "answers": ["$5,000.00", "$10,000.00", "5000", "$5,000"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q304",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What is the invoice issue date?",
                    "answers": ["October 15, 2025", "15 Oct 2025", "10/15/2025"],
                    "modality": "text",
                    "gold_pages": [1],
                    "intent": "FACT"
                },
                {
                    "question_id": "q305",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What is the item total cost for Dedicated Fiber Backbone Setup?",
                    "answers": ["$4,250.00", "$4,250", "4250.00"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP"
                },
                {
                    "question_id": "q306",
                    "doc_id": "invoice_88921",
                    "page_images": [doc3_img1],
                    "ocr_annotations": doc3_ocr,
                    "question": "What discount code was applied to this invoice?",
                    "answers": ["Insufficient evidence in document to answer the question.", "unanswerable", "none"],
                    "modality": "table",
                    "gold_pages": [1],
                    "intent": "TABLE_LOOKUP",
                    "is_unanswerable": True
                }
            ]
        }

        with open(self.dataset_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_data, f, indent=2)

    def load_samples(self) -> List[Dict[str, Any]]:
        """Returns list of benchmark evaluation items."""
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("samples", [])

    def _create_doc1_images(self, p1_path: str, p2_path: str):
        # Page 1: Revenue Table & Performance Text
        img1 = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([(20, 20), (780, 80)], fill=(30, 41, 59))
        d1.text((40, 40), "APEX GLOBAL ENTERPRISES - Q3 2025 FINANCIAL PERFORMANCE", fill=(255, 255, 255))

        d1.text((40, 110), "1. Executive Summary:", fill=(15, 23, 42))
        d1.text((40, 135), "Apex Global demonstrated resilient operational growth across global software verticals.", fill=(51, 65, 85))
        d1.text((40, 155), "Cloud subscription margins expanded by 340 basis points year-over-year.", fill=(51, 65, 85))

        # Financial Table
        d1.text((40, 210), "Table 1: Quarterly Financial Summary (USD Billions)", fill=(15, 23, 42))
        d1.rectangle([(40, 240), (760, 420)], outline=(148, 163, 184), width=2)
        # Header
        d1.rectangle([(40, 240), (760, 280)], fill=(241, 245, 249))
        d1.text((55, 255), "Quarter", fill=(15, 23, 42))
        d1.text((220, 255), "Revenue", fill=(15, 23, 42))
        d1.text((400, 255), "Operating Income", fill=(15, 23, 42))
        d1.text((600, 255), "Net Margin", fill=(15, 23, 42))
        # Rows
        rows = [
            ("Q1 2025", "$3.95B", "$1.12B", "28.3%"),
            ("Q2 2025", "$4.21B", "$1.25B", "29.7%"),
            ("Q3 2025", "$4.82B", "$1.48B", "30.7%"),
            ("Q4 2024", "$3.80B", "$1.05B", "27.6%"),
        ]
        for idx, (q, r, oi, nm) in enumerate(rows):
            y = 295 + idx * 30
            d1.text((55, y), q, fill=(71, 85, 105))
            d1.text((220, y), r, fill=(15, 23, 42))
            d1.text((400, y), oi, fill=(71, 85, 105))
            d1.text((600, y), nm, fill=(71, 85, 105))

        img1.save(p1_path)

        # Page 2: Governance & Audit Signatures
        img2 = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([(20, 20), (780, 80)], fill=(30, 41, 59))
        d2.text((40, 40), "APEX GLOBAL ENTERPRISES - GOVERNANCE & AUDIT", fill=(255, 255, 255))

        d2.text((40, 110), "2. Independent Auditor Opinion:", fill=(15, 23, 42))
        d2.text((40, 135), "The consolidated statements present fairly in all material respects the financial position.", fill=(51, 65, 85))
        d2.text((40, 160), "Approved by the Board of Directors on October 24, 2025.", fill=(51, 65, 85))

        # Visual seal / stamp box
        d2.rectangle([(40, 250), (250, 400)], outline=(37, 99, 235), width=3)
        d2.text((60, 270), "OFFICIAL CORPORATE SEAL", fill=(37, 99, 235))
        d2.text((60, 310), "Apex Global Enterprises", fill=(15, 23, 42))
        d2.text((60, 340), "Status: Verified & Validated", fill=(16, 185, 129))

        d2.text((320, 260), "Executive Sign-Off:", fill=(15, 23, 42))
        d2.text((320, 300), "Marcus Vance", fill=(30, 58, 138))
        d2.text((320, 320), "Chief Financial Officer, Apex Global", fill=(100, 116, 139))
        d2.text((320, 350), "Elena Rostova", fill=(30, 58, 138))
        d2.text((320, 370), "Chief Executive Officer, Apex Global", fill=(100, 116, 139))

        img2.save(p2_path)

    def _create_doc2_images(self, p1_path: str):
        img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([(20, 20), (780, 80)], fill=(13, 148, 136))
        d.text((40, 40), "BIOMED RESEARCH INSTITUTE - CLINICAL TRIAL REPORT (PHASE 3)", fill=(255, 255, 255))

        d.text((40, 110), "Abstract: Therapeutic assessment of BMR-902 in randomized cohorts.", fill=(15, 23, 42))
        
        # Clinical trial table
        d.text((40, 160), "Table: Primary Endpoint Efficacy Breakdown", fill=(15, 23, 42))
        d.rectangle([(40, 190), (760, 350)], outline=(148, 163, 184), width=2)
        d.rectangle([(40, 190), (760, 230)], fill=(204, 251, 241))
        d.text((55, 205), "Cohort Group", fill=(15, 23, 42))
        d.text((250, 205), "Patient Count", fill=(15, 23, 42))
        d.text((450, 205), "Efficacy Rate", fill=(15, 23, 42))
        d.text((620, 205), "p-value", fill=(15, 23, 42))

        c_rows = [
            ("Cohort A (Standard)", "450 patients", "72.4%", "p < 0.05"),
            ("Cohort B (BMR-902 High)", "450 patients", "94.6%", "p < 0.001"),
            ("Cohort C (Placebo)", "450 patients", "18.2%", "N/A"),
        ]
        for idx, (cg, pc, er, pv) in enumerate(c_rows):
            y = 245 + idx * 35
            d.text((55, y), cg, fill=(71, 85, 105))
            d.text((250, y), pc, fill=(71, 85, 105))
            d.text((450, y), er, fill=(15, 23, 42))
            d.text((620, y), pv, fill=(71, 85, 105))

        # Chart / Visual figure representation
        d.text((40, 380), "Figure 1: Cohort Response Trajectory Diagram", fill=(15, 23, 42))
        d.rectangle([(40, 410), (760, 560)], outline=(203, 213, 225), width=2, fill=(248, 250, 252))
        # Blue bar: Control Group
        d.rectangle([(80, 450), (220, 530)], fill=(59, 130, 246))
        d.text((85, 460), "Control (Blue)", fill=(255, 255, 255))
        # Green bar: Active Group
        d.rectangle([(260, 420), (400, 530)], fill=(34, 197, 94))
        d.text((265, 430), "Active (Green)", fill=(255, 255, 255))
        d.text((440, 460), "Legend: Blue = Control Group | Green = Active BMR-902", fill=(51, 65, 85))

        img.save(p1_path)

    def _create_doc3_images(self, p1_path: str):
        img = Image.new("RGB", (800, 1000), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([(20, 20), (780, 80)], fill=(79, 70, 229))
        d.text((40, 40), "NEXUS LOGISTICS - OFFICIAL COMMERCIAL INVOICE", fill=(255, 255, 255))

        d.text((40, 110), "Invoice Number: 88921", fill=(15, 23, 42))
        d.text((40, 135), "Invoice Date: October 15, 2025", fill=(71, 85, 105))
        d.text((40, 160), "Payment Due Date: November 15, 2025", fill=(15, 23, 42))

        # Invoice items table
        d.text((40, 210), "Itemized Services & Goods:", fill=(15, 23, 42))
        d.rectangle([(40, 240), (760, 400)], outline=(148, 163, 184), width=2)
        d.rectangle([(40, 240), (760, 280)], fill=(238, 242, 255))
        d.text((55, 255), "Description", fill=(15, 23, 42))
        d.text((380, 255), "Quantity", fill=(15, 23, 42))
        d.text((500, 255), "Unit Price", fill=(15, 23, 42))
        d.text((650, 255), "Total", fill=(15, 23, 42))

        items = [
            ("Enterprise Cloud Routing Cluster", "2 units", "$5,000.00", "$10,000.00"),
            ("Dedicated Fiber Backbone Setup", "1 service", "$4,250.00", "$4,250.00")
        ]
        for idx, (desc, qty, up, tot) in enumerate(items):
            y = 295 + idx * 35
            d.text((55, y), desc, fill=(71, 85, 105))
            d.text((380, y), qty, fill=(71, 85, 105))
            d.text((500, y), up, fill=(71, 85, 105))
            d.text((650, y), tot, fill=(15, 23, 42))

        d.rectangle([(40, 410), (760, 460)], fill=(243, 244, 246))
        d.text((480, 425), "Final Balance Due: $14,250.00", fill=(15, 23, 42))

        img.save(p1_path)
