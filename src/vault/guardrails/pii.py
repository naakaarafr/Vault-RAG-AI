import re

# PII Patterns
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b[6-9]\d{9}\b")
AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")

def redact_pii(text: str) -> str:
    """
    Redacts PII tokens (Aadhaar, PAN, Phone, Email) from the target text string.
    """
    if not text:
        return text

    # Redact Aadhaar first to prevent overlap with standard digits
    redacted = AADHAAR_REGEX.sub("[REDACTED_AADHAAR]", text)
    # Redact PAN
    redacted = PAN_REGEX.sub("[REDACTED_PAN]", redacted)
    # Redact Email
    redacted = EMAIL_REGEX.sub("[REDACTED_EMAIL]", redacted)
    # Redact Phone numbers
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)

    return redacted
