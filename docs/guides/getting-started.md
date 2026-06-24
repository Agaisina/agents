# Getting Started

## Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 – 3.13 |
| Poetry | ≥ 1.7 |
| Git | any |

## Installation

```bash
git clone https://github.com/your-org/agents.git
cd agents

# Initialise the virtual environment with the right Python
make venv

# Install all dependencies (runtime + dev)
make install
```

## Local model (Ollama)

If you want to run the system locally without a cloud LLM provider, install [Ollama](https://ollama.com) and pull the model configured in `settings.yaml`:

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model
ollama pull qwen2.5:7b

# Ollama starts automatically; verify it is running
ollama list
```

> The model used by the `local` environment is defined in `settings.yaml` under `local.models.model_id`. Change `qwen2.5:7b` there if you prefer a different model.

## Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```dotenv
# Active config profile (maps to a block in settings.yaml)
APP_ENV=local            # local | gemini

# LLM provider and API key — Ollama example (no key required)
LLM_PROVIDER=openai
APP_TOKEN=ollama
APP_BASE_URL=http://localhost:11434/v1

# Supabase vector store (RAG)
APP_DATABASE__SUPABASE_URL=https://<project-ref>.supabase.co
APP_DATABASE__SUPABASE_KEY=your_supabase_service_role_key

# LangGraph checkpointing (PostgreSQL / Supabase)
APP_DATABASE__POSTGRES_URI=postgresql://postgres.<ref>:<PASSWORD>@aws-0-eu-west-1.pooler.supabase.com:6543/postgres

# LangFuse observability (optional)
APP_LANGFUSE__ENABLED=true
APP_LANGFUSE__PUBLIC_KEY=pk-lf-xxx
APP_LANGFUSE__SECRET_KEY=sk-lf-xxx

# Frontend → backend URL
API_BASE_URL=https://<your-space>.hf.space
```

## Running the system

```bash
# Start the FastAPI server
poetry run uvicorn src.api.app:app --reload --port 8000

# Or start the Streamlit frontend
poetry run streamlit run src/frontend/app.py

# Or use the CLI
poetry run python -m src.cli.terminal
```

## Running tests

```bash
make test
```

## Building documentation

```bash
make docs       # build static site into site/
make docs-serve # build + live-reload server on :8001
```
