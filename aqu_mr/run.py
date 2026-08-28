"""
AQU-MR Command-Line Interface and Pipeline Runner
Supports document indexing, single-query execution with full --debug tracing,
benchmark evaluation, and web app launching.
"""

import sys
import os
import argparse
import json
import subprocess

# Ensure package root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.document_loader import DocumentLoader
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.generation.evidence_builder import EvidenceBuilder
from src.generation.answer_generator import GroundedAnswerGenerator
from src.generation.answer_validator import AnswerValidator
from src.evaluation.evaluate_qa import QAEvaluator
from src.evaluation.evaluate_retrieval import RetrievalEvaluator
from src.utils.logging_utils import setup_logger, DebugTracer


def run_query(doc_path: str, question: str, debug: bool = False, provider: str = "auto"):
    logger = setup_logger()
    tracer = DebugTracer()

    logger.info(f"Loading document: {doc_path}")
    loader = DocumentLoader()
    doc = loader.load_document(doc_path)
    logger.info(f"Loaded {doc.file_name} ({doc.total_pages} pages)")

    # 1. Query Understanding
    analyzer = QueryAnalyzer()
    q_contract = analyzer.analyze(question)
    tracer.log_event("Query Analysis", q_contract.raw_analysis)

    # 2. Modality-Aware Retrieval
    retriever = MultimodalRetriever()
    retriever.index_document(doc)
    selected_evidence, ranked_pages, fusion_debug = retriever.retrieve(question, q_contract)
    tracer.log_event("Retrieval & Fusion", fusion_debug)

    # 3. Evidence Assembly
    builder = EvidenceBuilder()
    evidence_packet = builder.build_packet(question, q_contract, selected_evidence)
    tracer.log_event("Evidence Assembly", {
        "pages_referenced": evidence_packet.pages_referenced,
        "total_items": evidence_packet.total_items
    })

    # 4. Grounded Generation
    generator = GroundedAnswerGenerator(provider=provider)
    answer = generator.generate(question, q_contract, evidence_packet)
    tracer.log_event("Answer Generation", {
        "answer_text": answer.answer_text,
        "confidence": answer.confidence,
        "source_pages": answer.source_pages,
        "reasoning": answer.reasoning_trace
    })

    # 5. Validation
    validator = AnswerValidator()
    val_res = validator.validate(answer, evidence_packet)
    tracer.log_event("Answer Validation", {
        "is_valid": val_res.is_valid,
        "grounding_score": val_res.grounding_score,
        "numerical_passed": val_res.numerical_passed,
        "entity_passed": val_res.entity_support_passed,
        "unsupported_claims": val_res.unsupported_claims
    })

    if debug:
        tracer.print_summary()
        print("\n" + "=" * 60)
        print("[EVIDENCE] Multimodal Evidence Packet Supplied to Generator")
        print("=" * 60)
        print(evidence_packet.raw_evidence_text)
        print("=" * 60 + "\n")

    print("\n" + "-" * 60)
    print(f"Question: {question}")
    print(f"Grounded Answer: {answer.answer_text}")
    print(f"Source Pages: {answer.source_pages}")
    print(f"Confidence: {answer.confidence:.1%}")
    print(f"Validation Status: {'PASSED' if val_res.is_valid else 'FLAGGED'} (Grounding Score: {val_res.grounding_score:.2f})")
    print(f"Reasoning Trace: {answer.reasoning_trace}")
    print("-" * 60 + "\n")


def run_benchmark(sample_limit: int = None):
    print("=" * 60)
    print("[BENCHMARK] AQU-MR Evaluation on DocVQA-2026")
    print("=" * 60)

    print("\n1. Evaluating Retrieval Modalities (Recall@K & MRR)...")
    r_evaluator = RetrievalEvaluator()
    r_res = r_evaluator.evaluate_benchmark(sample_limit=sample_limit)
    print(json.dumps(r_res, indent=2))

    print("\n2. Evaluating End-to-End Grounded QA (ANLS, EM, F1, Hallucination Rate)...")
    qa_evaluator = QAEvaluator()
    qa_res = qa_evaluator.evaluate_benchmark(sample_limit=sample_limit)
    
    # Clean output without long prediction dumps
    summary = {k: v for k, v in qa_res.items() if k != "sample_predictions"}
    print(json.dumps(summary, indent=2))

    print("\nSample Predictions Breakdown:")
    for p in qa_res.get("sample_predictions", [])[:5]:
        print(f"  • Q: {p['question'][:45]}... | Pred: {p['prediction']} | Gold: {p['gold_answers'][0]} | ANLS: {p['anls']}")


def main():
    parser = argparse.ArgumentParser(description="AQU-MR: Multimodal Document QA System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Query command
    query_p = subparsers.add_parser("query", help="Ask a question against a document")
    query_p.add_argument("--doc", required=True, help="Path to document (PDF, PNG, JPG, DOCX, TXT)")
    query_p.add_argument("--question", required=True, help="Natural-language question")
    query_p.add_argument("--debug", action="store_true", help="Print complete debug trace & raw evidence")
    query_p.add_argument("--provider", default="auto", choices=["auto", "gemini", "openai", "extractive"])

    # Eval command
    eval_p = subparsers.add_parser("eval", help="Run benchmark evaluation on DocVQA-2026")
    eval_p.add_argument("--samples", type=int, default=None, help="Limit number of evaluation samples")

    # App command
    app_p = subparsers.add_parser("app", help="Launch Streamlit Web UI")
    app_p.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")

    args = parser.parse_args()

    if args.command == "query":
        run_query(doc_path=args.doc, question=args.question, debug=args.debug, provider=args.provider)
    elif args.command == "eval":
        run_benchmark(sample_limit=args.samples)
    elif args.command == "app":
        app_file = os.path.join(os.path.dirname(__file__), "app", "app.py")
        subprocess.run(["streamlit", "run", app_file, "--server.port", str(args.port)])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
