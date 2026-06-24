# Veritas Corporate — Multi-Agent System

> **Automated corporate travel & expense compliance auditing powered by LangGraph.**

Veritas Corporate is a production-grade, multi-agent AI system that evaluates business-trip requests against the company's internal T&E policy documents in real time. It combines **RAG** (Retrieval-Augmented Generation), **structured LLM output**, **tool-calling loops**, and a **Human-in-the-Loop** approval workflow.

---

## Key features

| Feature | Detail |
|---|---|
| **Agent pipeline** | Researcher → Planner → Operator → Writer |
| **HITL approvals** | Manager interruption via LangGraph `interrupt_before` |
| **Multi-provider LLM** | OpenAI, Google Gemini — swap via `settings.yaml` |
| **Vector RAG** | ChromaDB (local) or Supabase + pgvector (cloud) |
| **Observability** | Full trace capture with LangFuse |
| **REST API** | FastAPI with SSE streaming and sliding-window rate limiting |
| **Frontend** | Streamlit UI with role simulation (Employee / Manager) |
| **Checkpointing** | SQLite (dev) or PostgreSQL / Supabase (prod) |

---

## Quick start

```bash
# 1. Install dependencies
make install

# 2. Copy and fill in your environment variables
cp .env.example .env          # add your API keys

# 3. Run the API server
poetry run uvicorn src.api.app:app --reload

# 4. Or launch the Streamlit frontend
poetry run streamlit run src/frontend/app.py

# 5. Build and serve the documentation
make docs
```

---

## Project layout

```
src/
├── agents/
│   ├── domain/          # Pure domain: interfaces, models, AgentConfig
│   ├── application/     # Factory, Registry, YAML prompts, use-cases
│   └── infrastructure/  # LLM adapters (Gemini / OpenAI / HF) + tools
├── workflow/
│   ├── graph.py         # LangGraph StateGraph definition
│   └── service.py       # AgentOrchestrator (sync + async)
├── api/                 # FastAPI application
├── frontend/            # Streamlit UI
├── infrastructure/      # VectorDB, Checkpointing, LangFuse monitor
└── config.py            # Dynaconf settings loader
data/
└── business_travel/     # Policy documents ingested into the vector store
docs/                    # This documentation
tests/                   # pytest test suite
```
