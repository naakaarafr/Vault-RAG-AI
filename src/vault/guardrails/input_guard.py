"""Input guardrails: PII redaction and prompt injection detection."""

import re
from dataclasses import dataclass


@dataclass
class InputCheckResult:
    """Status result from input guardrail verification."""

    is_safe: bool
    sanitized_text: str
    violations: list[str]


class InputGuardrail:
    """Guardrail enforcing input PII sanitization and prompt injection checks."""

    # Common PII regex patterns
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,T|api_key_[a-zA-Z0-9]{16,})\b",
    }

    # High-risk prompt injection signatures
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+prompt\s+override",
        r"you\s+are\n+now\s+a",
        r"reveal\s+(your\s+)?(secret|system\s+prompt|password)",
        r"disable\s+guardrails",
    ]

    def __init__(self, pii_redaction: bool = True, detect_injection: bool = True) -> None:
        self.pii_redaction = pii_redaction
        self.detect_injection = detect_injection

    def sanitize_and_check(self, text: str) -> InputCheckResult:
        """Inspect and sanitize input text."""
        violations: list[str] = []
        sanitized = text

        # 1. Prompt Injection Detection
        if self.detect_injection:
            for pattern in self.PROMPT_INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append(f"Prompt injection pattern detected: '{pattern}'")

        # 2. PII Redaction
        if self.pii_redaction:
            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, sanitized):
                    violations.append(f"PII type detected and redacted: {pii_type}")
                    sanitized = re.sub(pattern, f"[{pii_type}_REDACTED]", sanitized)

        is_safe = not any("Prompt injection" in v for v in violations)
        return InputCheckResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            violations=violations,
        )
