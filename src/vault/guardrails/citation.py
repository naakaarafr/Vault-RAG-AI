import re

CITATION_REGEX = re.compile(r"\[([a-zA-Z0-9_-]+)\]")

def extract_citations(text: str) -> list[str]:
    """
    Extracts all citation identifiers matching pattern [chunk_id].
    """
    if not text:
        return []
    return CITATION_REGEX.findall(text)

def validate_citations(answer: str, valid_chunk_ids: list[str]) -> tuple[bool, list[str]]:
    """
    Validates that the answer contains at least one citation referencing a valid chunk_id.
    Returns (is_valid, extracted_valid_citations).
    """
    extracted = extract_citations(answer)
    if not extracted:
        return False, []

    valid_set = set(valid_chunk_ids)
    matching = [cid for cid in extracted if cid in valid_set]

    if matching:
        return True, matching
    return False, []
