"""
AQU-MR: Multimodal Document Question Answering System
Adaptive Query Understanding and Modality-Aware Retrieval for Zero-Hallucination Document QA
Benchmark Dataset: VLR-CVC/DocVQA-2026
"""

import os
import sys
import tempfile
import json
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.document_loader import DocumentLoader
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.generation.evidence_builder import EvidenceBuilder
from src.generation.answer_generator import GroundedAnswerGenerator
from src.generation.answer_validator import AnswerValidator
from src.evaluation.docvqa_loader import DocVQABenchmarkLoader
from src.evaluation.evaluate_qa import QAEvaluator
from src.evaluation.evaluate_retrieval import RetrievalEvaluator
from src.evaluation.metrics import calculate_anls, calculate_exact_match, calculate_f1_score
from src.utils.schemas import ModalityType, Document, GroundedAnswer
from src.utils.question_generator import DocumentQuestionGenerator

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AQU-MR | Multimodal Document QA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS for professional enterprise design (Zero emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.2);
    }
    
    .answer-hero {
        background: #0f172a;
        color: #f8fafc;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 5px solid #0284c7;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .modality-badge {
        display: inline-block;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .modality-badge-table { background: #dcfce7; color: #166534; }
    .modality-badge-text { background: #dbeafe; color: #1e40af; }
    .modality-badge-visual { background: #fef3c7; color: #92400e; }
    .modality-badge-winner { background: #0284c7; color: #ffffff; }

    .provenance-tag {
        display: inline-block;
        background: #334155;
        color: #f8fafc;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    
    .validation-badge {
        display: inline-block;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
    }
    .validation-pass { background: #059669; color: #ffffff; }
    .validation-flag { background: #dc2626; color: #ffffff; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initializes and caches multimodal pipeline components."""
    loader = DocumentLoader()
    analyzer = QueryAnalyzer()
    retriever = MultimodalRetriever()
    builder = EvidenceBuilder()
    generator = GroundedAnswerGenerator()
    validator = AnswerValidator()
    benchmark_loader = DocVQABenchmarkLoader()
    return loader, analyzer, retriever, builder, generator, validator, benchmark_loader


def main():
    loader, analyzer, retriever, builder, generator, validator, benchmark_loader = get_pipeline()

    # Application Header
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-weight: 700; letter-spacing: -0.5px;">
            AQU-MR: Adaptive Query Understanding & Modality-Aware Retrieval
        </h2>
        <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 0.98rem;">
            Multimodal Document Question Answering System with Cross-Modal Answer Selection and Zero-Hallucination Grounding
        </p>
        <div style="margin-top: 10px; display:flex; gap:16px; font-size:0.82rem; opacity:0.85;">
            <span>Benchmark: <strong>VLR-CVC / DocVQA-2026</strong></span>
            <span>|</span>
            <span>Modalities: <strong>Text, Tables, Images</strong></span>
            <span>|</span>
            <span>Grounding: <strong>Strict Provenance Verification</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.header("Pipeline Configuration")
        
        provider = st.selectbox(
            "Generation Engine",
            ["Deterministic Extractive QA (Zero Hallucination)", "Google Gemini VLM", "OpenAI GPT-4o"],
            index=0,
            help="Extractive QA runs locally with zero external API calls. Gemini/OpenAI use API keys if provided."
        )
        resolved_provider = "extractive" if "Deterministic" in provider else ("gemini" if "Gemini" in provider else "openai")

        if resolved_provider == "gemini":
            api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key

        st.markdown("---")
        st.subheader("Modality Retrieval Priors")
        st.caption("AQU dynamically computes soft modality weights for every question:")
        st.markdown("- **Text Prior**: Hybrid BM25 and Dense Semantic Vectors")
        st.markdown("- **Table Prior**: 2D Cell Matrix and Coordinate Preservation")
        st.markdown("- **Image Prior**: Visual Embeddings and OCR Context")

        st.markdown("---")
        st.caption("Grounded Provenance: Answers strictly reference source pages, table cell coordinates, and visual layouts.")

    # Main Tab Navigation
    tab1, tab2, tab3 = st.tabs([
        "1. Training Dataset QA & Grounding Evaluation",
        "2. Unseen Document Multimodal QA & Selection",
        "3. System Architecture & Methods Reference"
    ])

    # =========================================================================
    # TAB 1: TRAINING DATASET QA & GROUNDING EVALUATION
    # =========================================================================
    with tab1:
        st.subheader("Training Dataset Questions & Grounding Answer Evaluation")
        st.markdown(
            "Select and execute question answering on official DocVQA-2026 benchmark training samples. "
            "Evaluates predictions against gold ground truth, verifies source provenance, and computes ANLS/F1 metrics."
        )

        all_benchmark_samples = benchmark_loader.load_samples()
        doc_categories = {
            "Apex Global Enterprises (Financial Report - Q3 2025)": "fin_report_2025",
            "BioMed Research Institute (Phase 3 Clinical Trial Report)": "clinical_study_09",
            "Nexus Logistics (Commercial Invoice 88921)": "invoice_88921"
        }

        col_t1, col_t2 = st.columns([1, 1.5])
        with col_t1:
            selected_doc_label = st.selectbox("Select Training Document:", list(doc_categories.keys()))
            selected_doc_id = doc_categories[selected_doc_label]

        # Filter samples for the selected training document
        doc_samples = [s for s in all_benchmark_samples if s["doc_id"] == selected_doc_id]
        sample_questions = [s["question"] for s in doc_samples]

        with col_t2:
            selected_train_q = st.selectbox("Select Training Question:", sample_questions)

        current_sample = next((s for s in doc_samples if s["question"] == selected_train_q), doc_samples[0])

        # Ingest document
        with st.spinner("Ingesting document modalities..."):
            train_doc = loader.from_docvqa_images(
                doc_id=selected_doc_id,
                page_images=current_sample["page_images"],
                ocr_annotations=current_sample.get("ocr_annotations")
            )
            retriever.index_document(train_doc)

        # Document overview expander
        with st.expander(f"Document Overview: {selected_doc_label} ({train_doc.total_pages} Pages)", expanded=False):
            page_cols = st.columns(max(len(train_doc.pages), 1))
            for p_idx, page in enumerate(train_doc.pages):
                with page_cols[p_idx]:
                    st.markdown(f"**Page {page.page_num}**")
                    if page.image_path and os.path.exists(page.image_path):
                        st.image(Image.open(page.image_path), use_container_width=True)
                    st.caption(f"Text Chunks: {len(page.text_chunks)} | Tables: {len(page.tables)} | Visuals: {len(page.images)}")

        if st.button("Evaluate Training Question with AQU-MR", type="primary", key="btn_train_eval"):
            with st.spinner("Analyzing question, retrieving cross-modal evidence, and validating grounding..."):
                # 1. Query Analysis
                q_contract = analyzer.analyze(selected_train_q)

                # 2. Retrieval & Fusion
                selected_evidence, ranked_pages, _ = retriever.retrieve(selected_train_q, q_contract)

                # 3. Evidence Packet
                packet = builder.build_packet(selected_train_q, q_contract, selected_evidence)

                # 4. Answer Generation
                generator.provider = resolved_provider
                train_ans = generator.generate(selected_train_q, q_contract, packet)

                # 5. Validation
                val_res = validator.validate(train_ans, packet)

                # 6. Evaluation metrics for this sample
                gold_answers = current_sample["answers"]
                sample_anls = calculate_anls(train_ans.answer_text, gold_answers)
                sample_em = calculate_exact_match(train_ans.answer_text, gold_answers)
                sample_f1 = calculate_f1_score(train_ans.answer_text, gold_answers)

            st.markdown("---")
            st.markdown("### Grounded Answer & Evaluation Metrics")
            
            # Grounded Answer Display
            ans_col1, ans_col2 = st.columns([2, 1])
            with ans_col1:
                provenance_str = " ".join([f'<span class="provenance-tag">Page {p}</span>' for p in train_ans.source_pages])
                val_class = "validation-pass" if val_res.is_valid else "validation-flag"
                val_text = f"Grounding Verified (Score: {val_res.grounding_score:.2f})" if val_res.is_valid else "Grounding Flagged"
                
                st.markdown(f"""
                <div class="answer-hero">
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 8px;">
                        Target Question: {selected_train_q}
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 600; color: #38bdf8; margin-bottom: 12px;">
                        {train_ans.answer_text}
                    </div>
                    <div style="display:flex; align-items:center; gap: 8px; margin-top: 10px;">
                        {provenance_str}
                        <span class="validation-badge {val_class}">{val_text}</span>
                        <span style="font-size: 0.85rem; opacity:0.85;">Confidence: {train_ans.confidence:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"Reasoning Trace: {train_ans.reasoning_trace}")

            with ans_col2:
                st.markdown("**Sample Grounding Evaluation:**")
                st.metric("ANLS Score", f"{sample_anls:.3f}")
                st.metric("Exact Match (EM)", f"{sample_em:.1%}")
                st.metric("Token F1 Score", f"{sample_f1:.3f}")
                st.caption(f"Gold Ground Truth: {gold_answers[0]}")

            # Evidence breakdown
            with st.expander("Supporting Evidence Details", expanded=False):
                st.text(packet.raw_evidence_text)

        st.markdown("---")
        st.subheader("Complete Training Benchmark Evaluation Suite")
        st.markdown("Run automated evaluation across all 18 benchmark samples to compute global metrics.")

        if st.button("Run Full Benchmark Suite Evaluation", key="btn_full_suite"):
            with st.spinner("Evaluating full benchmark suite across all document modalities..."):
                qa_eval = QAEvaluator()
                ret_eval = RetrievalEvaluator()
                qa_summary = qa_eval.evaluate_benchmark()
                ret_summary = ret_eval.evaluate_benchmark()

            # Global Metrics Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Overall ANLS Score", f"{qa_summary['ANLS']:.3f}")
            with m2:
                st.metric("Exact Match (EM)", f"{qa_summary['Exact_Match']:.1%}")
            with m3:
                st.metric("Hallucination Rate", f"{qa_summary['Hallucination_Rate']:.1%}")
            with m4:
                st.metric("Retrieval Recall@3", f"{ret_summary['Recall@3']:.1%}")

            # Modality Breakdown Table
            st.markdown("#### Modality-Level Performance Breakdown")
            mod_data = qa_summary.get("Modality_ANLS", {})
            m_df = pd.DataFrame([
                {"Modality": k.capitalize(), "ANLS Score": f"{v:.3f}", "Evaluation Metric": "DocVQA ANLS"}
                for k, v in mod_data.items()
            ])
            st.dataframe(m_df, use_container_width=True)

            # Detailed Sample Log Table
            st.markdown("#### Sample Predictions & Ground Truth Alignment")
            preds = qa_summary.get("sample_predictions", [])
            if preds:
                pred_df = pd.DataFrame(preds)[["question", "prediction", "gold_answers", "modality", "anls", "grounding_score", "is_valid"]]
                pred_df.columns = ["Question", "Predicted Answer", "Gold Ground Truth", "Modality", "ANLS", "Grounding Score", "Validation Status"]
                st.dataframe(pred_df, use_container_width=True)

    # =========================================================================
    # TAB 2: UNSEEN DOCUMENT MULTIMODAL QA & SELECTION
    # =========================================================================
    with tab2:
        st.subheader("Unseen Document Multimodal QA & Cross-Modality Selection")
        st.markdown(
            "Upload any unseen document (PDF, Word DOCX, Image, Text). "
            "AQU-MR extracts candidate answers from **Text**, **Table**, and **Visual** modalities, "
            "evaluates their grounding scores, and **selects the best modality candidate**."
        )

        unseen_file = st.file_uploader(
            "Upload Unseen Document (PDF, DOCX, PNG, JPG, TXT):",
            type=["pdf", "png", "jpg", "jpeg", "docx", "txt"],
            key="unseen_uploader"
        )

        active_unseen_doc: Optional[Document] = None

        if unseen_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{unseen_file.name}") as tmp:
                tmp.write(unseen_file.getvalue())
                tmp_path = tmp.name

            ext = os.path.splitext(unseen_file.name)[1].lower()
            with st.spinner(f"Ingesting {unseen_file.name}..."):
                if ext == ".pdf":
                    active_unseen_doc = loader.from_pdf(tmp_path)
                elif ext in [".docx", ".doc"]:
                    active_unseen_doc = loader.from_docx(tmp_path)
                elif ext in [".png", ".jpg", ".jpeg"]:
                    active_unseen_doc = loader.from_image(tmp_path)
                else:
                    active_unseen_doc = loader.from_text_file(tmp_path)

            retriever.index_document(active_unseen_doc)
            st.success(f"Successfully processed {unseen_file.name} ({active_unseen_doc.total_pages} Pages, {len(active_unseen_doc.pages[0].text_chunks)} Chunks, {len(active_unseen_doc.pages[0].tables)} Tables)")

            # Document Overview
            with st.expander(f"Document Overview: {unseen_file.name}", expanded=False):
                p_cols = st.columns(max(len(active_unseen_doc.pages), 1))
                for idx, p in enumerate(active_unseen_doc.pages):
                    with p_cols[idx]:
                        st.markdown(f"**Page {p.page_num}**")
                        if p.image_path and os.path.exists(p.image_path):
                            st.image(Image.open(p.image_path), use_container_width=True)
                        st.caption(f"Text Chunks: {len(p.text_chunks)} | Tables: {len(p.tables)} | Visuals: {len(p.images)}")

            # Dynamic Question Generator
            suggested_questions = DocumentQuestionGenerator.generate_questions(active_unseen_doc, max_questions=4)

            # Auto-reset question input if a new document is loaded
            if st.session_state.get("last_unseen_id") != active_unseen_doc.doc_id:
                st.session_state["last_unseen_id"] = active_unseen_doc.doc_id
                st.session_state["unseen_q_input"] = suggested_questions[0] if suggested_questions else ""

            def apply_unseen_question(q_text: str):
                st.session_state["unseen_q_input"] = q_text

            st.markdown("---")
            st.markdown("**Suggested Contextual Questions for this Document (Click to Select):**")
            s_cols = st.columns(min(len(suggested_questions), 4))
            for s_idx, sq in enumerate(suggested_questions[:4]):
                s_cols[s_idx].button(
                    sq,
                    key=f"unseen_btn_{active_unseen_doc.doc_id}_{s_idx}",
                    on_click=apply_unseen_question,
                    args=(sq,),
                    use_container_width=True
                )

            unseen_question = st.text_input(
                "Enter question regarding this document:",
                key="unseen_q_input"
            )

            if st.button("Answer Question & Compare Modalities", type="primary", key="btn_unseen_eval", use_container_width=True):
                if not unseen_question.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Analyzing question, retrieving cross-modal evidence, and comparing modality candidates..."):
                        # 1. Query Analysis
                        q_contract = analyzer.analyze(unseen_question)

                        # 2. Retrieval & Fusion
                        selected_evidence, ranked_pages, _ = retriever.retrieve(unseen_question, q_contract)

                        # 3. Evidence Packet
                        packet = builder.build_packet(unseen_question, q_contract, selected_evidence)

                        # 4. Grounded Generation & Cross-Modality Selection
                        generator.provider = resolved_provider
                        unseen_ans = generator.generate(unseen_question, q_contract, packet)

                        # 5. Validation
                        val_res = validator.validate(unseen_ans, packet)

                    st.markdown("---")

                    # Adaptive Query Understanding Summary
                    st.markdown("### Adaptive Query Understanding Contract")
                    qc1, qc2, qc3, qc4 = st.columns(4)
                    with qc1:
                        st.metric("Detected Intent", q_contract.intent.value)
                    with qc2:
                        st.metric("Expected Type", q_contract.expected_answer_type)
                    with qc3:
                        st.metric("Key Entities", ", ".join(q_contract.key_entities) if q_contract.key_entities else "None")
                    with qc4:
                        st.metric("Top Ranked Page", f"Page {ranked_pages[0]}" if ranked_pages else "N/A")

                    # Modality Priors
                    mw = q_contract.modality_weights
                    w1, w2, w3 = st.columns(3)
                    with w1:
                        st.progress(mw.get("text", 0.33), text=f"Text Weight: {mw.get('text', 0.33):.1%}")
                    with w2:
                        st.progress(mw.get("table", 0.33), text=f"Table Weight: {mw.get('table', 0.33):.1%}")
                    with w3:
                        st.progress(mw.get("image", 0.33), text=f"Visual Weight: {mw.get('image', 0.33):.1%}")

                    # Cross-Modality Comparison & Selection Matrix (User Req 3)
                    st.markdown("---")
                    st.markdown("### Cross-Modality Candidate Evaluation (Selection Matrix)")
                    st.caption("AQU-MR extracts candidate answers from each available modality and selects the highest scoring grounded candidate:")

                    modality_candidates = unseen_ans.metadata.get("modality_candidates", {})
                    best_modality = unseen_ans.metadata.get("best_modality", "text")
                    selection_rationale = unseen_ans.metadata.get("selection_rationale", "")

                    if modality_candidates:
                        cand_rows = []
                        for m_key, m_info in modality_candidates.items():
                            is_winner = (m_key == best_modality)
                            cand_rows.append({
                                "Modality": m_key.upper(),
                                "Candidate Answer": m_info["answer_text"],
                                "Extraction Confidence": f"{m_info['confidence']:.1%}",
                                "Modality Weight": f"{m_info['modality_weight']:.1%}",
                                "Weighted Score": f"{m_info['weighted_score']:.3f}",
                                "Source Pages": str(m_info["source_pages"]),
                                "Selection Status": "SELECTED (BEST MODALITY)" if is_winner else "Evaluated"
                            })
                        cand_df = pd.DataFrame(cand_rows)
                        st.dataframe(cand_df, use_container_width=True)

                        st.info(f"**Selection Rationale**: {selection_rationale}")
                    else:
                        st.warning("No modality candidate produced verifiable evidence. Clean controlled abstention triggered.")

                    # Winning Grounded Answer Hero Card
                    st.markdown("### Grounded Winning Answer")
                    prov_html = " ".join([f'<span class="provenance-tag">Page {p}</span>' for p in unseen_ans.source_pages])
                    val_class = "validation-pass" if val_res.is_valid else "validation-flag"
                    val_text = f"Grounding Verified (Score: {val_res.grounding_score:.2f})" if val_res.is_valid else "Grounding Flagged"
                    best_badge_html = f'<span class="modality-badge modality-badge-winner">BEST MODALITY: {best_modality.upper()}</span>'

                    st.markdown(f"""
                    <div class="answer-hero">
                        <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 8px;">
                            Question: {unseen_question}
                        </div>
                        <div style="font-size: 1.35rem; font-weight: 600; color: #38bdf8; margin-bottom: 12px;">
                            {unseen_ans.answer_text}
                        </div>
                        <div style="display:flex; align-items:center; gap: 8px; margin-top: 10px;">
                            {best_badge_html}
                            {prov_html}
                            <span class="validation-badge {val_class}">{val_text}</span>
                            <span style="font-size: 0.85rem; opacity:0.85;">Confidence: {unseen_ans.confidence:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"Reasoning Trace: {unseen_ans.reasoning_trace}")

                    # Live Unseen Metrics (User Req 4)
                    st.markdown("### Live Unseen Document Metrics")
                    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
                    with u_col1:
                        st.metric("Chosen Modality", best_modality.upper())
                    with u_col2:
                        st.metric("Grounding Score", f"{val_res.grounding_score:.2f}")
                    with u_col3:
                        st.metric("Confidence Level", f"{unseen_ans.confidence:.1%}")
                    with u_col4:
                        st.metric("Validation Status", "PASSED" if val_res.is_valid else "FLAGGED")

                    # Multimodal Evidence Inspector
                    st.markdown("### Selected Multimodal Evidence")
                    ev_tab1, ev_tab2, ev_tab3, ev_tab4 = st.tabs([
                        "Table Evidence", "Text Evidence", "Visual / Image Evidence", "Raw Evidence Packet"
                    ])

                    with ev_tab1:
                        tables_found = [it for it in selected_evidence if it.modality == ModalityType.TABLE]
                        if tables_found:
                            for t_it in tables_found:
                                st.markdown(f"**Page {t_it.page_num} - {t_it.evidence_id}**")
                                st.markdown(t_it.content)
                                if "matched_cells" in t_it.provenance_details and t_it.provenance_details["matched_cells"]:
                                    st.caption(f"Matched Cells: {t_it.provenance_details['matched_cells']}")
                        else:
                            st.caption("No table evidence retrieved.")

                    with ev_tab2:
                        text_found = [it for it in selected_evidence if it.modality == ModalityType.TEXT]
                        if text_found:
                            for tx_it in text_found:
                                st.markdown(f"**Page {tx_it.page_num} - {tx_it.evidence_id} (Score: {tx_it.score:.3f})**")
                                st.text(tx_it.content)
                        else:
                            st.caption("No text evidence retrieved.")

                    with ev_tab3:
                        img_found = [it for it in selected_evidence if it.modality == ModalityType.IMAGE]
                        if img_found:
                            for im_it in img_found:
                                st.markdown(f"**Page {im_it.page_num} - Visual Context**")
                                img_path = im_it.provenance_details.get("image_path")
                                if img_path and os.path.exists(img_path):
                                    st.image(Image.open(img_path), width=350)
                                st.caption(f"OCR Context: {im_it.provenance_details.get('ocr_snippet', '')}")
                        else:
                            st.caption("No visual evidence retrieved.")

                    with ev_tab4:
                        st.text(packet.raw_evidence_text)
        else:
            st.info("Upload a document above to begin multimodal question answering and cross-modality evaluation.")

    # =========================================================================
    # TAB 3: SYSTEM ARCHITECTURE & METHODS REFERENCE
    # =========================================================================
    with tab3:
        st.subheader("AQU-MR System Architecture & Methodology Reference")
        st.markdown("""
        ### Adaptive Query Understanding and Modality-Aware Retrieval
        
        The **AQU-MR** system implements a decoupled 8-stage architecture for robust, zero-hallucination document question answering:

        1. **Dual Ingestion Engine**:
           - **Benchmark Ingestion**: Image-first ingestion with word-level OCR token coordinates and bounding boxes.
           - **Unseen Ingestion**: Universal document parser for PDF, DOCX, TXT, and images.
        
        2. **First-Class Modality Processing**:
           - **Text**: Conservative text cleaning and semantic paragraph/sentence chunking preserving character offsets.
           - **Tables**: Preserves 2D row/column matrices, headers, and exact cell coordinates without flattening.
           - **Images**: Dual-signal representation combining visual embeddings with page-level OCR context.
        
        3. **Adaptive Query Understanding (AQU)**:
           - Classifies query intent: `TABLE_LOOKUP`, `FACT`, `VISUAL`, `COMPARISON`, `LIST`, `SUMMARY`.
           - Assigns soft, continuous modality routing weights: $(w_{\\text{text}}, w_{\\text{table}}, w_{\\text{image}})$.
        
        4. **Modality-Aware Hybrid Retrieval**:
           - Text: Hybrid BM25 lexical + dense TF-IDF vector similarity.
           - Tables: Structured header and cell coordinate matching.
           - Images: Visual feature alignment + OCR keyword matching.
        
        5. **Cross-Modality Candidate Selection**:
           - Extracts candidate answers across Text, Table, and Visual modalities.
           - Evaluates confidence scores, weights them by query intent, and selects the optimal grounded candidate.
        
        6. **Structured Evidence Assembly**:
           - Packages cross-modal evidence separated by page and modality within strict token budgets.
        
        7. **Grounded Zero-Hallucination Generation**:
           - Extractive QA or VLM generation restricted strictly to verifiable evidence facts, numbers, and dates.
        
        8. **Rigorous Provenance Validation**:
           - Verifies entity support, numerical precision, table coordinates, and enforces controlled abstention.
        """)


if __name__ == "__main__":
    main()
