.PHONY: up down test lint benchmark

up:
	docker compose up -d

down:
	docker compose down

test:
	python -m pytest -v

lint:
	ruff check .

benchmark:
	python scripts/benchmark_rag.py
