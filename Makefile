# AORA-Forge developer commands. All targets are CPU-only and run offline (mock
# LLM, hashing embedder) unless ANTHROPIC_API_KEY is set, in which case `demo`
# hits the real Claude API.

.PHONY: help install dev test lint format typecheck check demo hook-demo dashboard clean

help:
	@echo "make install     - pip install the package"
	@echo "make dev         - install with dev + embed extras"
	@echo "make test        - run pytest"
	@echo "make lint        - ruff check"
	@echo "make format      - ruff format"
	@echo "make typecheck   - mypy"
	@echo "make check       - lint + typecheck + test (what CI runs)"
	@echo "make demo        - end-to-end growth demo (real API if ANTHROPIC_API_KEY set, else mock)"
	@echo "make hook-demo   - prove the planner gains tools after growth"
	@echo "make dashboard   - launch the visual dashboard (needs: pip install -e .[viz])"
	@echo "make clean       - remove caches and demo stores"

install:
	pip install -e .

dev:
	pip install -e ".[dev,embed]"

test:
	pytest tests/ -q

lint:
	ruff check aora_forge scripts tests

format:
	ruff format aora_forge scripts tests

typecheck:
	mypy aora_forge

check: lint typecheck test
	@ruff format --check aora_forge scripts tests

demo:
	python scripts/demo_full_loop.py

hook-demo:
	python scripts/orchestrator_hook_demo.py --mock

dashboard:
	python scripts/dashboard.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ \
	       demo_skill_store hook_demo_store clusters.jsonl
	find . -name '*.pyc' -delete
