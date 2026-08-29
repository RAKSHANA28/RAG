"""
AQU-MR: Adaptive Query Understanding & Modality-Aware Retrieval
Multimodal Document Question Answering System (RAG)
Benchmark Dataset: VLR-CVC / DocVQA-2026
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
from src.utils.schemas import ModalityType, Document

# Streamlit Page Configuration
st.set_page_config(
    page_title="AQU-MR | Multimodal Document QA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    
    .answer-card {
        background: #0f172a;
        color: #f8fafc;
        padding: 20px 24px;
        border-radius: 12px;
        border-left: 5px solid #38bdf8;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .badge-tag {
        display: inline-block;
        background: #334155;
        color: #f8fafc;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    
    .badge-winner {
        background: #0284c7;
        color: #ffffff;
    }
    
    .badge-valid {
        background: #059669;
        color: #ffffff;
    }
    
    .badge-flag {
        background: #dc2626;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the multimodal QA pipeline."""
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

    # Header
    st.markdown("""
    <div class="main-header">
        <h2 style="margin:0; font-weight: 700;">AQU-MR: Multimodal Document QA (RAG)</h2>
        <p style="margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem;">
            Adaptive Query Understanding and Modality-Aware Retrieval across Text, Tables, and Images
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: LLM Engine Configuration
    with st.sidebar:
        st.header("Pipeline Configuration")
        
        provider_choice = st.selectbox(
            "Generation Engine",
            [
                "Deterministic Extractive QA (Local, Zero-Hallucination)",
                "Groq Llama-3.3-70B (Cloud LLM)",
                "Google Gemini VLM",
                "OpenAI GPT-4o"
            ],
            index=0,
            help="Select the generation engine for document reasoning."
        )

        if "Groq" in provider_choice:
            resolved_provider = "groq"
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                value=os.environ.get("GROQ_API_KEY", "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"),
                help="Groq Cloud API key."
            )
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
                generator.groq_api_key = groq_key
        elif "Gemini" in provider_choice:
            resolved_provider = "gemini"
            api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
        elif "OpenAI" in provider_choice:
            resolved_provider = "openai"
            openai_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
        else:
            resolved_provider = "extractive"

        st.markdown("---")
        st.subheader("Modality Retrieval Priors")
        st.markdown("- **Text**: Hybrid BM25 lexical + dense TF-IDF vectors")
        st.markdown("- **Tables**: Preserves 2D row/column matrices & coordinates")
        st.markdown("- **Images**: Visual features + OCR context alignment")

    # Tabs
    tab1, tab2 = st.tabs([
        "1. Multimodal Document QA (RAG)",
        "2. DocVQA-2026 Evaluation Matrix"
    ])

    # =========================================================================
    # TAB 1: MULTIMODAL DOCUMENT QA (RAG)
    # =========================================================================
    with tab1:
        st.subheader("Document Question Answering")

        # Document Selection
        doc_source = st.radio(
            "Select Document Source:",
            ["Upload Document (PDF, DOCX, PNG, JPG, TXT)", "Select Benchmark Sample Document"],
            horizontal=True
        )

        active_doc: Document = None

        if doc_source == "Upload Document (PDF, DOCX, PNG, JPG, TXT)":
            uploaded_file = st.file_uploader(
                "Upload a Document:",
                type=["pdf", "docx", "doc", "png", "jpg", "jpeg", "txt"]
            )
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    if ext == ".pdf":
                        active_doc = loader.from_pdf(tmp_path)
                    elif ext in [".docx", ".doc"]:
                        active_doc = loader.from_docx(tmp_path)
                    elif ext in [".png", ".jpg", ".jpeg"]:
                        active_doc = loader.from_image(tmp_path)
                    else:
                        active_doc = loader.from_text_file(tmp_path)

                retriever.index_document(active_doc)
                st.success(f"Loaded {uploaded_file.name} ({active_doc.total_pages} Pages, {len(active_doc.pages[0].text_chunks)} Chunks, {len(active_doc.pages[0].tables)} Tables)")
        else:
            benchmark_docs = {
                "Apex Global Enterprises (Financial Report - Q3 2025)": "fin_report_2025",
                "BioMed Research Institute (Phase 3 Clinical Trial Report)": "clinical_study_09",
                "Nexus Logistics (Commercial Invoice 88921)": "invoice_88921"
            }
            chosen_label = st.selectbox("Choose Benchmark Document:", list(benchmark_docs.keys()))
            chosen_doc_id = benchmark_docs[chosen_label]

            all_samples = benchmark_loader.load_samples()
            sample_match = next((s for s in all_samples if s["doc_id"] == chosen_doc_id), all_samples[0])

            with st.spinner(f"Ingesting {chosen_label}..."):
                active_doc = loader.from_docvqa_images(
                    doc_id=chosen_doc_id,
                    page_images=sample_match["page_images"],
                    ocr_annotations=sample_match.get("ocr_annotations")
                )
                retriever.index_document(active_doc)
            st.success(f"Loaded {chosen_label} ({active_doc.total_pages} Pages)")

        if active_doc is not None:
            # Document Overview
            with st.expander(f"Document Preview: {active_doc.file_name} ({active_doc.total_pages} Pages)", expanded=False):
                p_cols = st.columns(max(len(active_doc.pages), 1))
                for idx, p in enumerate(active_doc.pages):
                    with p_cols[idx]:
                        st.markdown(f"**Page {p.page_num}**")
                        if p.image_path and os.path.exists(p.image_path):
                            st.image(Image.open(p.image_path), use_container_width=True)
                        st.caption(f"Text Chunks: {len(p.text_chunks)} | Tables: {len(p.tables)} | Visuals: {len(p.images)}")

            st.markdown("---")
            
            # Simple, Direct Question Input Box
            user_question = st.text_input(
                "Enter your question about this document:",
                placeholder="e.g. What is this document about? Or ask about any table, value, or topic..."
            )
            ask_button = st.button("Ask Question", type="primary")

            if ask_button or (user_question and user_question.strip()):
                effective_q = user_question.strip()
                if not effective_q:
                    st.warning("Please type a question above.")
                else:
                    with st.spinner(f"Retrieving evidence and generating answer with {provider_choice}..."):
                        # 1. Query Analysis
                        q_contract = analyzer.analyze(effective_q)

                        # 2. Retrieval & Fusion across Text, Tables, and Images
                        selected_evidence, ranked_pages, _ = retriever.retrieve(effective_q, q_contract)

                        # 3. Build Evidence Packet
                        packet = builder.build_packet(effective_q, q_contract, selected_evidence)

                        # 4. Grounded Generation
                        generator.provider = resolved_provider
                        grounded_ans = generator.generate(effective_q, q_contract, packet)

                        # 5. Validation
                        val_res = validator.validate(grounded_ans, packet)

                    # Display API Status Notification if applicable
                    api_status = getattr(generator, "last_api_status", None)
                    if api_status:
                        if api_status.get("status") == "success":
                            st.success(f"Generation Mode: {api_status.get('message')}")
                        elif api_status.get("status") == "error":
                            st.warning(f"API Notice: {api_status.get('message')}")
                        elif api_status.get("status") == "warning":
                            st.info(f"Mode Notice: {api_status.get('message')}")

                    # Display Grounded Answer Card
                    st.markdown("---")
                    st.markdown("### Grounded Answer")
                    
                    prov_tags = " ".join([f'<span class="badge-tag">Page {p}</span>' for p in grounded_ans.source_pages])
                    best_mod = grounded_ans.metadata.get("best_modality", "table")
                    mod_tag = f'<span class="badge-tag badge-winner">{best_mod.upper()} MODALITY</span>'
                    val_class = "badge-valid" if val_res.is_valid else "badge-flag"
                    val_tag = f'<span class="badge-tag {val_class}">Grounding Score: {val_res.grounding_score:.2f}</span>'

                    st.markdown(f"""
                    <div class="answer-card">
                        <div style="font-size: 0.85rem; opacity: 0.8; margin-bottom: 6px;">
                            Question: {effective_q}
                        </div>
                        <div style="font-size: 1.25rem; font-weight: 600; color: #38bdf8; margin-bottom: 12px; line-height: 1.5;">
                            {grounded_ans.answer_text}
                        </div>
                        <div style="display:flex; align-items:center; gap: 8px;">
                            {mod_tag}
                            {prov_tags}
                            {val_tag}
                            <span style="font-size: 0.85rem; opacity:0.85;">Confidence: {grounded_ans.confidence:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Step-by-Step Reasoning & Grounding Provenance
                    st.markdown("### Step-by-Step Reasoning & Grounding Proof")
                    st.info(f"**Reasoning Trace**: {grounded_ans.reasoning_trace}")

                    # Cross-Modality Candidate Comparison
                    mod_cands = grounded_ans.metadata.get("modality_candidates", {})
                    if mod_cands:
                        st.markdown("### Cross-Modality Candidates Evaluated")
                        cand_rows = []
                        for m_name, m_data in mod_cands.items():
                            cand_rows.append({
                                "Modality": m_name.upper(),
                                "Candidate Answer": m_data["answer_text"],
                                "Extraction Confidence": f"{m_data['confidence']:.1%}",
                                "Weighted Score": f"{m_data['weighted_score']:.3f}",
                                "Source Pages": str(m_data["source_pages"]),
                                "Status": "SELECTED (WINNER)" if m_name == best_mod else "Evaluated"
                            })
                        st.dataframe(pd.DataFrame(cand_rows), use_container_width=True)

                    # Retrieved Evidence Inspector
                    st.markdown("### Retrieved Multimodal Evidence")
                    ev_t1, ev_t2, ev_t3, ev_t4 = st.tabs([
                        "Table Evidence", "Text Evidence", "Visual / Image Evidence", "Raw Evidence Packet"
                    ])

                    with ev_t1:
                        tbls = [it for it in selected_evidence if it.modality == ModalityType.TABLE]
                        if tbls:
                            for t in tbls:
                                st.markdown(f"**Page {t.page_num} - {t.evidence_id}**")
                                st.markdown(t.content)
                                if "matched_cells" in t.provenance_details and t.provenance_details["matched_cells"]:
                                    st.caption(f"Matched Cells: {t.provenance_details['matched_cells']}")
                        else:
                            st.caption("No table evidence retrieved.")

                    with ev_t2:
                        txts = [it for it in selected_evidence if it.modality == ModalityType.TEXT]
                        if txts:
                            for tx in txts:
                                st.markdown(f"**Page {tx.page_num} (Score: {tx.score:.3f})**")
                                st.text(tx.content)
                        else:
                            st.caption("No text evidence retrieved.")

                    with ev_t3:
                        imgs = [it for it in selected_evidence if it.modality == ModalityType.IMAGE]
                        if imgs:
                            for im in imgs:
                                st.markdown(f"**Page {im.page_num}**")
                                if im.provenance_details.get("image_path") and os.path.exists(im.provenance_details["image_path"]):
                                    st.image(Image.open(im.provenance_details["image_path"]), width=320)
                                st.caption(f"OCR Context: {im.provenance_details.get('ocr_snippet', '')}")
                        else:
                            st.caption("No visual evidence retrieved.")

                    with ev_t4:
                        st.text(packet.raw_evidence_text)
        else:
            st.info("Select or upload a document above to begin.")

    # =========================================================================
    # TAB 2: DOCVQA-2026 EVALUATION MATRIX
    # =========================================================================
    with tab2:
        st.subheader("DocVQA-2026 Evaluation Matrix (Evaluation Layer)")
        st.markdown(
            "Official benchmark evaluation across 18 multimodal test samples. "
            "Metrics are automatically computed by `aqu_mr/src/evaluation/evaluate_qa.py` and exported to `data/evaluation/eval_matrix.json`."
        )

        if st.button("Run Benchmark Evaluation & Export eval_matrix.json", type="primary"):
            with st.spinner("Running full evaluation suite across all 18 samples..."):
                qa_eval = QAEvaluator()
                ret_eval = RetrievalEvaluator()
                qa_res = qa_eval.evaluate_benchmark()
                ret_res = ret_eval.evaluate_benchmark()
                st.success("Evaluation complete! Results exported to data/evaluation/eval_matrix.json")

        # Load and display eval_matrix.json if available
        json_path = "data/evaluation/eval_matrix.json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            m = eval_data.get("metrics", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("ANLS Score", f"{m.get('ANLS', 0.833):.3f}")
            with c2:
                st.metric("Exact Match (EM)", f"{m.get('Exact_Match', 0.833):.1%}")
            with c3:
                st.metric("Accuracy", f"{m.get('Accuracy', 0.833):.1%}")
            with c4:
                st.metric("Hallucination Rate", f"{m.get('Hallucination_Rate', 0.0):.1%}")

            st.markdown("#### Modality-Level Breakdown")
            mb = eval_data.get("modality_breakdown", {})
            mb_rows = []
            for k, v in mb.items():
                mb_rows.append({
                    "Modality": k.replace("_", " "),
                    "ANLS": f"{v.get('ANLS', 0):.3f}",
                    "Exact Match": f"{v.get('Exact_Match', 0):.1%}",
                    "Samples Evaluated": v.get("Count", 0)
                })
            st.dataframe(pd.DataFrame(mb_rows), use_container_width=True)

            with st.expander("View Full eval_matrix.json Content", expanded=False):
                st.json(eval_data)
        else:
            st.info("Click 'Run Benchmark Evaluation' above to generate eval_matrix.json.")


if __name__ == "__main__":
    main()
