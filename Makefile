.DEFAULT_GOAL := help
.PHONY: help install bootstrap run test lint typecheck coverage clean docker-build docker-up

PYTHON := python3
VENV   := .venv
BIN    := $(VENV)/bin

help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create venv & install dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

bootstrap: ## Generate synthetic data + train anomaly model (ONNX)
	$(BIN)/python scripts/generate_synthetic_data.py
	$(BIN)/python scripts/train_anomaly_model.py
	$(BIN)/python scripts/seed_db.py

run: ## Run the API locally on :8000
	$(BIN)/uvicorn sentinel.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run the pytest suite
	$(BIN)/pytest -v

lint: ## Ruff lint + format check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

typecheck: ## mypy strict
	$(BIN)/mypy src

coverage: ## Coverage report
	$(BIN)/pytest --cov=src/sentinel --cov-report=term-missing --cov-report=html

clean: ## Remove build artifacts
	rm -rf $(VENV) build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build: ## Build the production image
	docker build -t loanbook-sentinel:latest .

docker-up: ## Bring up the full dev stack (API + Postgres + Prom + Grafana)
	docker compose up -d --build
