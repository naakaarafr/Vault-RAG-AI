"""Output guardrails: Groundedness metric verification and format validation."""

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class OutputCheckResult:
    """Status result from output guardrail verification."""

    is_grounded: bool
    groundedness_score: float
    violations: list[str]


class OutputGuardrail:
    """Guardrail enforcing answer groundedness and JSON schema validity."""

    def __init__(self, min_groundedness_score: float = 0.3) -> None:
        self.min_groundedness_score = min_groundedness_score

    def check_groundedness(self, answer: str, context: str) -> OutputCheckResult:
        """Calculate lexical overlap groundedness score of answer against retrieved context."""
        violations: list[str] = []
        if not answer.strip() or not context.strip():
            return OutputCheckResult(is_grounded=True, groundedness_score=1.0, violations=[])

        # Ignore common stop words for token comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or",
            "in", "on", "at", "to", "for", "of", "with", "by", "it",
            "this", "that",
        }

        answer_tokens = set(re.findall(r"\b\w+\b", answer.lower())) - stop_words
        context_tokens = set(re.findall(r"\b\w+\b", context.lower())) - stop_words

        if not answer_tokens:
            return OutputCheckResult(is_grounded=True, groundedness_score=1.0, violations=[])

        # Proportion of answer non-stopword tokens present in context
        supported_tokens = answer_tokens.intersection(context_tokens)
        score = len(supported_tokens) / len(answer_tokens)

        is_grounded = score >= self.min_groundedness_score
        if not is_grounded:
            violations.append(
                f"Groundedness score {score:.2f} below threshold {self.min_groundedness_score}"
            )

        return OutputCheckResult(
            is_grounded=is_grounded,
            groundedness_score=score,
            violations=violations,
        )

    @staticmethod
    def validate_json_schema(text: str) -> tuple[bool, dict[str, Any] | None]:
        """Validate whether LLM output string is valid JSON."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return True, parsed
            return False, None
        except (json.JSONDecodeError, TypeError):
            return False, None
