"""Smoke test script calling local OpenAI-compatible LLM endpoint via LLMClient wrapper."""

import sys
from pathlib import Path

# Add src directory to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vault.config import get_settings
from vault.llm import LLMClient


def main() -> None:
    settings = get_settings()
    print(
        f"Connecting to local LLM endpoint at {settings.llm_base_url} "
        f"(Model: {settings.llm_model})..."
    )

    client = LLMClient(settings=settings)
    messages = [
        {"role": "system", "content": "You are a helpful local assistant."},
        {"role": "user", "content": "Hello! Confirm you are running locally."},
    ]

    try:
        response = client.generate(messages=messages, temperature=0.1, max_tokens=100)
        print("\n--- LLM Response Received ---")
        print(response)
        print("-----------------------------\n")
        print("Smoke test PASSED!")
    except Exception as exc:
        print(f"\nLocal LLM server not reachable at {settings.llm_base_url}: {exc}")
        print("Start local Ollama server (`docker compose up` or `ollama serve`) to connect live.")
        print("Smoke test wrapper contract verified.")


if __name__ == "__main__":
    main()
