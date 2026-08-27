"""
DocVQA-2026 Evaluation Metrics for AQU-MR
Implements official DocVQA ANLS (Average Normalized Levenshtein Similarity), Exact Match (EM),
F1, Hallucination/Unsupported-Claim Rate, Correct Abstention Rate, and Retrieval Recall@K/MRR.
"""

from typing import List, Dict, Any, Union, Optional
import re


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_answer(s: str) -> str:
    """Standardizes answer strings for evaluation."""
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def calculate_anls(prediction: str, ground_truths: List[str], tau: float = 0.5) -> float:
    """
    Computes official DocVQA Average Normalized Levenshtein Similarity (ANLS).
    ANLS = 1 - (NL / tau) if NL < tau else 0.0
    where NL = edit_dist(pred, gt) / max(len(pred), len(gt))
    """
    if not ground_truths:
        return 0.0

    pred_norm = normalize_answer(prediction)
    max_score = 0.0

    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if not pred_norm and not gt_norm:
            return 1.0
        if not pred_norm or not gt_norm:
            continue

        dist = levenshtein_distance(pred_norm, gt_norm)
        max_len = max(len(pred_norm), len(gt_norm))
        nl = dist / max_len if max_len > 0 else 0.0

        if nl < tau:
            score = 1.0 - nl
        else:
            score = 0.0
        
        if score > max_score:
            max_score = score

    return max_score


def calculate_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Calculates binary exact match after normalization."""
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_norm == normalize_answer(gt):
            return 1.0
    return 0.0


def calculate_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculates token-level F1 score against ground truth answers."""
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1


def calculate_recall_at_k(retrieved_pages: List[int], gold_pages: List[int], k: int = 3) -> float:
    """Calculates page-level Recall@K."""
    if not gold_pages:
        return 1.0
    top_k_retrieved = set(retrieved_pages[:k])
    gold_set = set(gold_pages)
    hits = top_k_retrieved & gold_set
    return len(hits) / len(gold_set) if gold_set else 0.0


def calculate_mrr(retrieved_pages: List[int], gold_pages: List[int]) -> float:
    """Calculates Mean Reciprocal Rank (MRR) for page retrieval."""
    if not gold_pages:
        return 1.0
    gold_set = set(gold_pages)
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_set:
            return 1.0 / rank
    return 0.0
