"""
Adaptive Query Understanding (AQU) Engine
Derives dynamic QueryContracts with intent classification, soft modality confidence routing,
named entity extraction, and answer type expectation.
"""

from typing import Dict, Any, List, Optional
import re
from ..utils.schemas import QueryContract, QueryIntent


class QueryAnalyzer:
    """Analyzes incoming natural-language queries without hardcoded question overrides."""

    # Lexical patterns indicating intent & modality affinity
    TABLE_KEYWORDS = {
        "table", "row", "column", "col", "cell", "total", "sum", "average",
        "rate", "percentage", "%", "revenue", "sales", "cost", "budget", "quarter",
        "q1", "q2", "q3", "q4", "growth", "vs", "versus", "compare", "breakdown",
        "highest", "lowest", "max", "min", "rank", "tabular", "net", "gross"
    }

    VISUAL_KEYWORDS = {
        "image", "photo", "picture", "figure", "fig", "diagram", "chart", "graph",
        "plot", "logo", "color", "red", "blue", "green", "signature", "stamp",
        "box", "circle", "arrow", "top left", "bottom right", "icon", "visual",
        "header", "banner", "layout", "font", "stamp", "seal"
    }

    COMPARISON_KEYWORDS = {
        "compare", "difference", "between", "versus", "vs", "higher than",
        "lower than", "increase", "decrease", "change from", "exceed", "growth rate"
    }

    LIST_KEYWORDS = {
        "list", "all", "every", "name the", "what are the", "items", "categories",
        "products", "names of", "types of"
    }

    TEMPORAL_KEYWORDS = {
        "year", "month", "date", "when", "quarter", "annual", "period", "duration",
        "199", "200", "201", "202", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december"
    }

    def analyze(self, query: str) -> QueryContract:
        """
        Derives a complete QueryContract for soft modality-aware routing.
        """
        cleaned_query = query.strip()
        lower_q = cleaned_query.lower()
        words = set(re.findall(r'\b\w+\b', lower_q))

        # 1. Intent Detection
        intent = self._detect_intent(lower_q, words)

        # 2. Soft Modality Confidence Scoring
        modality_weights = self._compute_modality_weights(lower_q, words, intent)

        # 3. Entity & Key Term Extraction
        key_entities = self._extract_entities(cleaned_query)

        # 4. Expected Answer Type Detection
        answer_type = self._detect_expected_answer_type(lower_q, words)

        # 5. Requirement flags
        requires_comp = bool(words & self.COMPARISON_KEYWORDS)
        requires_temp = bool(words & self.TEMPORAL_KEYWORDS) or bool(re.search(r'\b(19|20)\d{2}\b', lower_q))

        return QueryContract(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            modality_weights=modality_weights,
            key_entities=key_entities,
            expected_answer_type=answer_type,
            requires_comparison=requires_comp,
            requires_temporal=requires_temp,
            raw_analysis={
                "detected_intent": intent.value,
                "modality_scores": modality_weights,
                "extracted_entities": key_entities,
                "answer_type": answer_type
            }
        )

    def _detect_intent(self, lower_q: str, words: set) -> QueryIntent:
        if words & self.COMPARISON_KEYWORDS or "compare" in lower_q:
            return QueryIntent.COMPARISON
        elif words & self.LIST_KEYWORDS and ("what are" in lower_q or "list" in lower_q):
            return QueryIntent.LIST
        elif len(words & self.VISUAL_KEYWORDS) >= 1 or any(k in lower_q for k in ["color", "look like", "signature", "diagram", "figure", "seal", "stamp", "logo"]):
            return QueryIntent.VISUAL
        elif len(words & self.TABLE_KEYWORDS) >= 2 or any(k in lower_q for k in ["how much", "total", "percentage", "table", "invoice", "balance", "price"]):
            return QueryIntent.TABLE_LOOKUP
        elif any(lower_q.startswith(w) for w in ["summarize", "overview", "what is this document about"]):
            return QueryIntent.SUMMARY
        return QueryIntent.FACT

    def _compute_modality_weights(self, lower_q: str, words: set, intent: QueryIntent) -> Dict[str, float]:
        """
        Computes soft, normalized modality weights (text, table, image).
        Never sets a modality to zero so all modalities can contribute.
        """
        # Baseline prior
        text_score = 1.0
        table_score = 0.8
        image_score = 0.6

        # Keyword overlaps
        table_overlap = len(words & self.TABLE_KEYWORDS)
        visual_overlap = len(words & self.VISUAL_KEYWORDS)

        table_score += table_overlap * 0.4
        image_score += visual_overlap * 0.45

        # Intent adjustments
        if intent == QueryIntent.TABLE_LOOKUP:
            table_score += 1.2
            text_score += 0.3
        elif intent == QueryIntent.VISUAL:
            image_score += 1.4
            text_score += 0.2
        elif intent == QueryIntent.COMPARISON:
            table_score += 0.8
            text_score += 0.5
        elif intent == QueryIntent.SUMMARY:
            text_score += 1.0
            image_score += 0.3
        elif intent == QueryIntent.FACT:
            text_score += 0.8

        # Numerical queries often rely heavily on tables or formatted text
        if re.search(r'\b(how much|how many|amount|price|cost|\$|€|£|%|number)\b', lower_q):
            table_score += 0.7

        total = text_score + table_score + image_score
        return {
            "text": round(text_score / total, 3),
            "table": round(table_score / total, 3),
            "image": round(image_score / total, 3)
        }

    def _extract_entities(self, query: str) -> List[str]:
        """Extracts candidate entities, capitalized phrases, quoted phrases, and numeric tokens."""
        entities = []
        
        # 1. Quoted terms
        quotes = re.findall(r'["\'](.*?)["\']', query)
        entities.extend(quotes)

        # 2. Capitalized phrases (excluding start of sentence)
        words = query.split()
        for i, w in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w and clean_w[0].isupper() and (i > 0 or len(clean_w) > 3):
                if clean_w.lower() not in {"what", "where", "when", "which", "who", "whose", "how", "the", "and", "in"}:
                    entities.append(clean_w)

        # 3. Numeric tokens, years, codes (e.g. "2024", "Q3", "10%", "$500M")
        num_tokens = re.findall(r'\b(?:Q[1-4]|20\d{2}|19\d{2}|\d+(?:\.\d+)?%?|\$\d+[\w]*)\b', query, re.IGNORECASE)
        entities.extend(num_tokens)

        # Deduplicate preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            e_norm = e.strip().lower()
            if e_norm and e_norm not in seen:
                seen.add(e_norm)
                unique_entities.append(e.strip())

        return unique_entities

    def _detect_expected_answer_type(self, lower_q: str, words: set) -> str:
        if lower_q.startswith("when") or "what date" in lower_q or "which year" in lower_q:
            return "DATE"
        elif any(lower_q.startswith(w) for w in ["how much", "how many", "what percentage", "what is the rate", "what total"]):
            return "NUMBER"
        elif any(lower_q.startswith(w) for w in ["who", "what organization", "which company", "which author", "which person"]):
            return "ENTITY"
        elif lower_q.startswith("is ") or lower_q.startswith("does ") or lower_q.startswith("did ") or lower_q.startswith("was "):
            return "BOOLEAN"
        elif "list" in lower_q or "what are" in lower_q:
            return "LIST"
        return "TEXT"
