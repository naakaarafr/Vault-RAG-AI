"""Corpus manifest file downloader utility for Vault."""

import csv
import sys
import urllib.request
from pathlib import Path

# Add src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def download_corpus(manifest_path: str = "corpus/manifest.csv") -> None:
    """Download source documents listed in manifest.csv."""
    path = Path(manifest_path)
    if not path.exists():
        print(f"Manifest file not found at {manifest_path}. Please create it first.")
        return

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id")
            source_url = row.get("source_url")
            file_path = row.get("file_path")

            if not file_path:
                continue

            target_file = Path(file_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)

            if source_url and source_url.startswith("http"):
                print(f"Downloading [{doc_id}] from {source_url} -> {file_path}...")
                try:
                    urllib.request.urlretrieve(source_url, target_file)
                    print(f"Successfully downloaded [{doc_id}].")
                except Exception as exc:
                    print(f"Failed to download [{doc_id}] from {source_url}: {exc}")
            else:
                print(f"Skipping download for [{doc_id}]: No valid HTTP source_url provided.")


if __name__ == "__main__":
    manifest = sys.argv[1] if len(sys.argv) > 1 else "corpus/manifest.csv"
    download_corpus(manifest)
