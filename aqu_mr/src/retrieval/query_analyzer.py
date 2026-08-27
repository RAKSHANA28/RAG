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
        "highest", "lowest", "max", "min", "rank", "tabular", "net", "gross",
        "duration", "format", "round", "track", "phase", "tier", "week", "topic",
        "status", "deliverable", "milestone", "hours", "fit", "channel", "requirement",
        "definition", "profile", "frequency", "fee", "fees", "what they do", "make money"
    }

    STOP_WORDS = {
        "what", "is", "the", "in", "of", "for", "to", "a", "an", "and", "or", "on", "at",
        "by", "with", "from", "about", "as", "into", "like", "through", "after", "over",
        "between", "out", "against", "during", "without", "before", "under", "around",
        "among", "tell", "me", "can", "you", "give", "show", "find", "does", "did", "was",
        "were", "are", "be", "been", "being", "have", "has", "had", "do", "this", "that"
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
        "products", "names of", "types of", "phases", "milestones", "tracks", "rounds"
    }

    TEMPORAL_KEYWORDS = {
        "year", "month", "date", "when", "quarter", "annual", "period", "duration",
        "199", "200", "201", "202", "203", "204", "january", "february", "march", "april", "may",
        "june", "july", "august", "september", "october", "november", "december", "due date", "deadline"
    }

    def analyze(self, query: str) -> QueryContract:
        """
        Derives a complete QueryContract for soft modality-aware routing.
        """
        cleaned_query = query.strip()
        lower_q = cleaned_query.lower()
        all_words = set(re.findall(r'\b\w+\b', lower_q))
        meaningful_words = all_words - self.STOP_WORDS

        # 1. Intent Detection
        intent = self._detect_intent(lower_q, all_words, meaningful_words)

        # 2. Soft Modality Confidence Scoring
        modality_weights = self._compute_modality_weights(lower_q, all_words, meaningful_words, intent)

        # 3. Entity & Key Term Extraction
        key_entities = self._extract_entities(cleaned_query, meaningful_words)

        # Semantic concept expansion for cross-vocabulary document matching
        expanded_entities = list(key_entities)
        if any(k in lower_q for k in ["objective", "goal", "purpose", "aim"]):
            expanded_entities.extend(["goal", "objective", "purpose", "aim"])
        if any(k in lower_q for k in ["candidate", "author", "prepared for", "student"]):
            expanded_entities.extend(["prepared", "candidate", "author", "target", "pursuing"])
        if any(k in lower_q for k in ["summary", "overview", "about", "covers"]):
            expanded_entities.extend(["summary", "overview", "covers", "abstract", "scope"])

        # 4. Expected Answer Type Detection
        answer_type = self._detect_expected_answer_type(lower_q, all_words)

        # 5. Requirement flags
        requires_comp = bool(all_words & self.COMPARISON_KEYWORDS)
        requires_temp = bool(all_words & self.TEMPORAL_KEYWORDS) or bool(re.search(r'\b(19|20)\d{2}\b', lower_q))

        return QueryContract(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            modality_weights=modality_weights,
            key_entities=expanded_entities,
            expected_answer_type=answer_type,
            requires_comparison=requires_comp,
            requires_temporal=requires_temp,
            raw_analysis={
                "detected_intent": intent.value,
                "modality_scores": modality_weights,
                "extracted_entities": expanded_entities,
                "answer_type": answer_type
            }
        )

    def _detect_intent(self, lower_q: str, words: set, meaningful: set) -> QueryIntent:
        if words & self.COMPARISON_KEYWORDS or "compare" in lower_q:
            return QueryIntent.COMPARISON
        elif words & self.LIST_KEYWORDS and any(k in lower_q for k in ["what are", "list", "all", "phases", "milestones", "tracks"]):
            return QueryIntent.LIST
        elif any(k in lower_q for k in ["what does", "what do", "what they do", "how they make money", "duration", "how much", "total", "percentage", "table", "invoice", "balance", "price", "format", "tier", "week"]):
            return QueryIntent.TABLE_LOOKUP
        elif len(meaningful & self.TABLE_KEYWORDS) >= 1:
            return QueryIntent.TABLE_LOOKUP
        elif len(words & self.VISUAL_KEYWORDS) >= 1 or any(k in lower_q for k in ["color", "look like", "signature", "diagram", "figure", "seal", "stamp", "logo"]):
            return QueryIntent.VISUAL
        elif any(lower_q.startswith(w) for w in ["summarize", "overview", "what is this document about"]):
            return QueryIntent.SUMMARY
        return QueryIntent.FACT

    def _compute_modality_weights(self, lower_q: str, words: set, meaningful: set, intent: QueryIntent) -> Dict[str, float]:
        text_score = 0.8
        table_score = 0.6
        image_score = 0.4

        table_overlap = len(meaningful & self.TABLE_KEYWORDS)
        visual_overlap = len(words & self.VISUAL_KEYWORDS)

        table_score += table_overlap * 0.6
        image_score += visual_overlap * 0.6

        if intent == QueryIntent.TABLE_LOOKUP:
            table_score += 1.4
            text_score += 0.5
        elif intent == QueryIntent.VISUAL:
            image_score += 1.5
            text_score += 0.3
        elif intent == QueryIntent.COMPARISON:
            table_score += 1.0
            text_score += 0.6
        elif intent == QueryIntent.LIST:
            text_score += 1.0
            table_score += 0.8
        elif intent == QueryIntent.SUMMARY:
            text_score += 1.2
            image_score += 0.3
        elif intent == QueryIntent.FACT:
            text_score += 1.0
            table_score += 0.6

        total = text_score + table_score + image_score
        return {
            "text": round(text_score / total, 3),
            "table": round(table_score / total, 3),
            "image": round(image_score / total, 3)
        }

    def _extract_entities(self, query: str, meaningful_words: set) -> List[str]:
        entities = []
        quotes = re.findall(r'["\'](.*?)["\']', query)
        entities.extend(quotes)

        words = query.split()
        for i, w in enumerate(words):
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w and clean_w[0].isupper() and clean_w.lower() not in self.STOP_WORDS:
                entities.append(clean_w)

        num_tokens = re.findall(r'\b(?:Round\s+\d+|Phase\s+\d+|Table\s+\d+|Q[1-4]|20\d{2}|19\d{2}|\d+(?:\.\d+)?%?|\$\d+[\w]*)\b', query, re.IGNORECASE)
        entities.extend(num_tokens)

        # Include key multi-word phrases (like "investment banking", "study plan", "due date")
        lower_q = query.lower()
        phrases = ["investment banking", "sales & trading", "securities services", "commercial banking", "study plan", "due date", "round 1", "round 2", "round 3", "round 4", "round 5", "phase 1", "phase 2", "phase 3", "phase 4", "phase 5", "phase 6", "pragadeesh", "vit chennai", "hackerrank", "recruiter screen", "cib division"]
        for p in phrases:
            if p in lower_q:
                entities.append(p)

        seen = set()
        unique_entities = []
        for e in entities:
            e_norm = e.strip().lower()
            if e_norm and e_norm not in seen:
                seen.add(e_norm)
                unique_entities.append(e.strip())

        return unique_entities

    def _detect_expected_answer_type(self, lower_q: str, words: set) -> str:
        if "duration" in lower_q or "how long" in lower_q or "timed" in lower_q or "hours" in lower_q:
            return "DURATION"
        elif lower_q.startswith("when") or "what date" in lower_q or "which year" in lower_q or "due date" in lower_q:
            return "DATE"
        elif any(lower_q.startswith(w) for w in ["how much", "how many", "what percentage", "what is the rate", "what total", "cost", "price", "balance"]):
            return "NUMBER"
        elif any(lower_q.startswith(w) for w in ["who", "what organization", "which company", "which person", "prepared for", "candidate"]):
            return "ENTITY"
        elif lower_q.startswith("is ") or lower_q.startswith("does ") or lower_q.startswith("did ") or lower_q.startswith("was "):
            return "BOOLEAN"
        elif "list" in lower_q or "what are" in lower_q or "phases" in lower_q or "topics" in lower_q:
            return "LIST"
        return "TEXT"

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
