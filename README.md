---
title: Veritas Corporate — Multi-Agent System
emoji: ✈️
colorFrom: indigo
colorTo: purple
sdk: docker
sdk_version: 6.9.0
pinned: false
license: mit
short_description: Multi-agent T&E compliance auditing with LangGraph
---

# ✈️ Veritas Corporate — Multi-Agent System

> **Automated corporate travel & expense compliance auditing powered by LangGraph.**

A production-grade multi-agent AI system that evaluates business-trip requests against internal T&E policy documents in real time. Combines RAG, structured LLM output, tool-calling loops, and a Human-in-the-Loop approval workflow.

---

## Architecture

```
Employee request
      │
      ▼
 🔍 Researcher  ──►  RAGFacts  (policy limits extracted from vector store)
      │
      ▼
 📅 Planner     ──►  PlannerFacts  (itinerary + geo tools)
      │
      ▼
 ⚖️  Operator   ──►  OperatorFacts  (TCO audit + compliance verdict)
      │
      ├── AUTO_APPROVED ──────────────────────────────┐
      ├── NEEDS_REVIEW  ──►  👨‍💼 Manager (HITL) ──────┤
      └── REJECTED      ──►  back to Planner          │
                                                       ▼
                                                  ✍️ Writer
                                                       │
                                              final_response
```

**Providers supported:** OpenAI · Google Gemini  
**Vector stores:** ChromaDB (local) · Supabase + pgvector (cloud)  
**Checkpointing:** SQLite (dev) · PostgreSQL / Supabase (prod)

---

## Quickstart

```bash
# Install
git clone https://github.com/your-org/agents.git && cd agents
make install

# Configure
cp .env.example .env   # fill in your API keys

# Run the API
poetry run uvicorn src.api.app:app --reload

# Or the Streamlit frontend
poetry run streamlit run src/frontend/app.py

# Or the CLI
poetry run python -m src.cli.terminal
```

## Tests

```bash
make test
```

## Documentation

```bash
make docs        # build static site → site/
make docs-serve  # live-reload on http://127.0.0.1:8001
```

---

## Project structure

```
src/
├── agents/
│   ├── domain/          # IAgent · AgentConfig · Pydantic models
│   ├── application/     # Factory · Registry · YAML prompts
│   └── infrastructure/  # LLM adapters (Gemini / OpenAI) + tools
├── workflow/
│   ├── graph.py         # LangGraph StateGraph
│   └── service.py       # AgentOrchestrator (sync + async)
├── api/                 # FastAPI  — /v1/chat  /v1/chat/stream  /v1/chat/resume
├── frontend/            # Streamlit UI (Employee + Manager roles)
├── infrastructure/      # VectorDB · Checkpointing · LangFuse monitor
└── config.py            # Dynaconf settings loader
data/
└── business_travel/     # T&E policy documents (RAG knowledge base)
docs/                    # MkDocs documentation
tests/                   # pytest
```

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/chat` | Synchronous query |
| `POST` | `/v1/chat/stream` | SSE streaming (NDJSON) |
| `POST` | `/v1/chat/resume` | Resume after manager approval |

Interactive docs at `http://localhost:8000/docs` when the server is running.
