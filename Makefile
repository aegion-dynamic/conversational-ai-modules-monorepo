# Makefile-style commands for the project
# Supports both Poetry and uv workflows

.PHONY: help install install-poetry install-uv sync test lint format clean setup-uv

help: ## Show this help message
	@echo "Available commands:"
	@echo "  install-poetry  - Install dependencies using Poetry (traditional)"
	@echo "  install-uv      - Install dependencies using uv (fast)"
	@echo "  install         - Install using uv if available, fallback to Poetry"
	@echo "  sync            - Sync uv with Poetry lock file"
	@echo "  test            - Run tests"
	@echo "  lint            - Run linting with pylint"
	@echo "  format          - Format code with black and isort"
	@echo "  clean           - Clean build artifacts"
	@echo "  setup-uv        - Set up uv alongside Poetry"

install-poetry: ## Install dependencies using Poetry
	poetry install

install-uv: ## Install dependencies using uv (requires sync first)
	@if command -v uv >/dev/null 2>&1; then \
		uv pip install -r requirements.txt; \
	else \
		echo "uv not found. Installing..."; \
		pip install uv; \
		uv pip install -r requirements.txt; \
	fi

install: ## Install using uv if available, fallback to Poetry
	@if command -v uv >/dev/null 2>&1 && [ -f requirements.txt ]; then \
		echo "Using uv for fast installation..."; \
		uv pip install -r requirements.txt; \
	else \
		echo "Using Poetry for installation..."; \
		poetry install; \
	fi

sync: ## Export Poetry dependencies and prepare for uv
	poetry export -f requirements.txt --output requirements.txt --with dev
	@echo "Dependencies exported to requirements.txt for uv usage"

test: ## Run tests
	@if command -v uv >/dev/null 2>&1; then \
		uv run pytest; \
	else \
		poetry run pytest; \
	fi

lint: ## Run linting
	@if command -v uv >/dev/null 2>&1; then \
		uv run pylint nlqs/ expert_system/ tot/ utils/ tog/ state_machine/; \
	else \
		poetry run pylint nlqs/ expert_system/ tot/ utils/ tog/ state_machine/; \
	fi

format: ## Format code
	@if command -v uv >/dev/null 2>&1; then \
		uv run black .; \
		uv run isort .; \
	else \
		poetry run black .; \
		poetry run isort .; \
	fi

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ 2>/dev/null || true

setup-uv: ## Set up uv alongside Poetry
	@if command -v pwsh >/dev/null 2>&1; then \
		pwsh -ExecutionPolicy Bypass -File setup_uv.ps1; \
	else \
		python setup_uv.py; \
	fi
