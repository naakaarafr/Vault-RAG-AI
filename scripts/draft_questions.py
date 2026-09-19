"""Draft candidate Q&A pairs using local LLM and export to eval/review.csv."""

import csv
import json
import sys
from pathlib import Path

# Add src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vault.config import get_settings
from vault.llm import LLMClient
from vault.vector_store import VectorStore


def draft_candidate_questions() -> None:
    """Generate candidate Q&A pairs from ingested document chunks."""
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    vector_db_file = data_dir / "vector_db.json"

    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    review_csv_path = eval_dir / "review.csv"

    if not vector_db_file.exists():
        print(f"No vector database found at {vector_db_file}. Please run ingestion first.")
        return

    store = VectorStore(persistence_path=vector_db_file)
    if not store.documents:
        print("Vector database is empty. Add documents before drafting questions.")
        return

    llm = LLMClient(settings=settings)
    rows: list[dict[str, str]] = []
    qid_counter = 1

    print(f"Drafting candidate questions from {len(store.documents)} chunks...")

    for doc_id, doc in store.documents.items():
        prompt = (
            "Given the following document text, create 1 specific fact-based question and answer.\n"
            f"Document Text:\n{doc.content[:600]}\n\n"
            "Format your output as JSON with keys 'question' and 'gold_answer'."
        )

        try:
            response = llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=256,
            )
            # Try to parse JSON output
            data = json.loads(response)
            default_q = f"What does {doc.metadata.get('title', 'document')} discuss?"
            question = data.get("question", default_q)
            gold_answer = data.get("gold_answer", doc.content[:100])
        except Exception:
            # Fallback if local server offline/formatting error
            title_text = doc.metadata.get("title", doc_id)
            question = f"What is described in section of {title_text}?"
            gold_answer = doc.content[:120]

        rows.append({
            "qid": f"Q_{qid_counter:03d}",
            "question": question,
            "gold_doc_ids": doc.metadata.get("doc_id", doc_id),
            "gold_answer": gold_answer,
            "answerable": "true",
            "verified": "false",  # Requires human verification in review.csv
        })
        qid_counter += 1

    # Write review.csv
    fieldnames = ["qid", "question", "gold_doc_ids", "gold_answer", "answerable", "verified"]
    with open(review_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDrafted {len(rows)} candidate questions written to {review_csv_path}.")
    print("Human Verification Step: Set 'verified' column to 'true' for approved rows.")


if __name__ == "__main__":
    draft_candidate_questions()
