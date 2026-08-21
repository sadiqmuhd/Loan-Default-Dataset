.PHONY: help install contract train test lint types fmt serve clean report notebooks

help:
	@echo "install   Install the package and dev dependencies"
	@echo "contract  Regenerate config/data_contract.yaml from the dataset"
	@echo "train     Train, calibrate and register a new model version"
	@echo "test      Run the full test suite with coverage"
	@echo "lint      ruff check + format check"
	@echo "types     mypy type check"
	@echo "fmt       Apply ruff formatting"
	@echo "serve     Run the API locally with reload"
	@echo "report    Regenerate the validation and fairness reports"
	@echo "notebooks Rebuild and execute the analysis notebooks"
	@echo ""
	@echo "Deploy: push to GitHub; Railway builds via nixpacks.toml and serves"
	@echo "        uvicorn loan_default.api.main:app on \$$PORT."

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

contract:
	python scripts/generate_data_contract.py

train:
	python -m loan_default.models.train

test:
	pytest tests/ -v --cov=loan_default --cov-report=term-missing

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

types:
	mypy

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

serve:
	uvicorn loan_default.api.main:app --reload --host 0.0.0.0 --port 8000

notebooks:
	python scripts/build_notebooks.py

report:
	python scripts/validation_report.py
	python scripts/fairness_report.py

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
