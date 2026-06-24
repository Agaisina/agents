TOP := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PACKAGE_DIR := $(TOP)

# Search the for a compatible Python version, starting with 3.13
PYTHON_EXEC := $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)

.PHONY: help venv install test shell clean docs docs-serve docs-deploy api-local frontend-local

help:
	@echo "make venv          # tell poetry to initialize/use the correct python environment"
	@echo "make install       # install dependencies + package via poetry"
	@echo "make test          # run pytest inside the poetry environment"
	@echo "make shell         # spawn a shell within the virtual environment"
	@echo "make clean         # remove the virtual environment and cache files"
	@echo "make docs          # build the MkDocs documentation site into site/"
	@echo "make docs-serve    # build and serve docs with live-reload on :8001"
	@echo "make docs-deploy   # deploy docs to GitHub Pages (gh-pages branch)"
	@echo "make api-local     # start FastAPI backend with local Ollama model (port 8000)"
	@echo "make frontend-local# start Streamlit frontend pointing to local API (port 8000)"

venv:
	@echo "Found Python executable: $(PYTHON_EXEC)"
	poetry env use $(PYTHON_EXEC)

install:
	poetry install --no-root

test:
	poetry run pytest -q $(PACKAGE_DIR)/tests

shell:
	poetry shell

clean:
	poetry env remove --all

docs:
	poetry run mkdocs build --strict

docs-serve:
	poetry run mkdocs serve --dev-addr 127.0.0.1:8001

docs-deploy:
	poetry run mkdocs gh-deploy --force
	@rm -rf **/__pycache__ **/*.pyc .pytest_cache

api-local:
	APP_ENV=local APP_EXECUTION_MODE=api poetry run python -m src.main

frontend-local:
	API_BASE_URL=http://localhost:8000 poetry run streamlit run src/frontend/app.py