"""Entry point for python -m vault.retrieve CLI execution."""

import argparse
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vault.retrieve.cli import run_retrieval_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Vault Retrieval CLI")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of hits to return (default: 5)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval mode (default: hybrid_rerank)",
    )
    parser.add_argument(
        "--role",
        type=str,
        action="append",
        dest="roles",
        help="Caller role(s) for RBAC filter (e.g. --role public --role compliance)",
    )

    args = parser.parse_args()
    roles = args.roles or ["public"]

    run_retrieval_query(
        query=args.query,
        k=args.top_k,
        mode=args.mode,
        roles=roles,
    )


if __name__ == "__main__":
    main()
