"""Ingestion pipeline CLI execution logic with file hash tracking for idempotency."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from vault.config import get_settings
from vault.ingest.chunker import StructureAwareChunker
from vault.ingest.indexer import IngestionIndexer
from vault.ingest.loaders import DocxLoader, PDFLoader


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change tracking."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_ingestion(
    manifest_path: str = "corpus/manifest.csv",
    force: bool = False,
) -> dict[str, Any]:
    """Execute manifest document ingestion, chunking, and indexing."""
    manifest = Path(manifest_path)
    if not manifest.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        return {"status": "error", "message": "Manifest not found"}

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    hash_record_file = data_dir / "ingestion_hashes.json"

    # Load existing hashes for idempotency
    existing_hashes: dict[str, str] = {}
    if hash_record_file.exists() and not force:
        try:
            with open(hash_record_file, encoding="utf-8") as f:
                existing_hashes = json.load(f)
        except Exception:
            existing_hashes = {}

    pdf_loader = PDFLoader()
    docx_loader = DocxLoader()
    chunker = StructureAwareChunker()
    indexer = IngestionIndexer(settings=settings)

    manifest_rows: list[dict[str, Any]] = []
    total_docs_processed = 0
    total_pages_processed = 0
    total_chunks_indexed = 0
    skipped_docs = 0

    with open(manifest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id", "").strip()
            title = row.get("title", doc_id).strip()
            file_path_str = row.get("file_path", "").strip()
            raw_roles = row.get("allowed_roles", "public").split(";")
            allowed_roles = [r.strip() for r in raw_roles if r.strip()]

            manifest_rows.append(row)

            if not file_path_str:
                continue

            file_path = Path(file_path_str)
            if not file_path.exists():
                print(f"[Ingest] Skipping [{doc_id}]: File not found at '{file_path_str}'")
                continue

            # Check hash for idempotency
            file_hash = compute_file_hash(file_path)
            if doc_id in existing_hashes and existing_hashes[doc_id] == file_hash and not force:
                print(f"[Ingest] Skipping [{doc_id}]: Unchanged file content (Hash match)")
                skipped_docs += 1
                continue

            print(f"[Ingest] Processing [{doc_id}] '{title}' ({file_path.name})...")

            # Load document
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                pages = pdf_loader.load(file_path)
            elif ext in (".docx", ".doc"):
                pages = docx_loader.load(file_path)
            else:
                print(f"[Ingest] Unsupported file format '{ext}' for [{doc_id}]")
                continue

            # Chunk document
            chunks = chunker.chunk_document(
                doc_id=doc_id,
                title=title,
                allowed_roles=allowed_roles,
                pages=pages,
            )

            # Index chunks
            indexer.index_chunks(chunks)

            total_docs_processed += 1
            total_pages_processed += len(pages)
            total_chunks_indexed += len(chunks)
            existing_hashes[doc_id] = file_hash

    # Register Postgres metadata
    indexer.register_circular_index_postgres(manifest_rows)

    # Save updated hashes
    with open(hash_record_file, "w", encoding="utf-8") as f:
        json.dump(existing_hashes, f, indent=2)

    summary = {
        "status": "success",
        "processed_documents": total_docs_processed,
        "skipped_documents": skipped_docs,
        "total_pages": total_pages_processed,
        "total_chunks": total_chunks_indexed,
    }

    # Print clean summary report
    print("\n================ Ingestion Summary ================")
    print(f" Documents Processed : {total_docs_processed}")
    print(f" Documents Skipped   : {skipped_docs}")
    print(f" Total Pages Read    : {total_pages_processed}")
    print(f" Total Chunks Created: {total_chunks_indexed}")
    print("====================================================\n")

    return summary
