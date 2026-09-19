"""Entry point for python -m vault.ingest CLI execution."""

import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vault.ingest.cli import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Vault Ingestion Pipeline CLI")
    parser.add_argument(
        "--manifest",
        type=str,
        default="corpus/manifest.csv",
        help="Path to manifest.csv file (default: corpus/manifest.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of unchanged documents",
    )
    args = parser.parse_args()

    result = run_ingestion(manifest_path=args.manifest, force=args.force)
    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
