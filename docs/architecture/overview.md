# Architecture Overview

## Clean Architecture layers

```mermaid
graph TB
    subgraph PRES["🖥️  Presentation Layer"]
        direction LR
        API["⚡ FastAPI\n/v1/chat"]
        FE["🎨 Streamlit\nFrontend"]
        CLI["💻 CLI\nTerminal"]
    end

    subgraph WORK["⚙️  Workflow Layer"]
        direction LR
        GRAPH["🔀 LangGraph\nStateGraph"]
        SVC["🎯 AgentOrchestrator\nservice.py"]
    end

    subgraph APP["📦  Application Layer"]
        direction LR
        FAC["🏭 Factory"]
        REG["📋 Registry"]
        PROMPTS["📝 YAML Prompts"]
    end

    subgraph DOM["💎  Domain Layer"]
        direction LR
        IFACE["🔷 IAgent\nInterface"]
        CFG["⚙️ AgentConfig"]
        MODELS["📐 Pydantic\nModels"]
    end

    subgraph INFRA["🔧  Infrastructure Layer"]
        direction LR
        GEMINI["🌟 Gemini\nAdapter"]
        OAI["🤖 OpenAI\nAdapter"]
        VDB["🗄️ VectorDB"]
        CP["💾 Checkpointer"]
        LF["📊 LangFuse"]
    end

    PRES --> WORK
    WORK --> APP
    APP --> DOM
    APP --> INFRA
    DOM --> INFRA

    classDef presStyle fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef workStyle fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef appStyle fill:#cffafe,stroke:#06b6d4,color:#164e63
    classDef domStyle fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef infraStyle fill:#fef9c3,stroke:#eab308,color:#713f12

    class API,FE,CLI presStyle
    class GRAPH,SVC workStyle
    class FAC,REG,PROMPTS appStyle
    class IFACE,CFG,MODELS domStyle
    class GEMINI,OAI,HF,VDB,CP,LF infraStyle
```

## Component diagram

```mermaid
graph TB
    subgraph UI["User Interfaces"]
        FE["🎨 Streamlit\nFrontend"]
        CLI2["💻 CLI"]
    end

    subgraph BACKEND["Backend"]
        API2["⚡ FastAPI\n/v1/chat/stream"]
        ORCH["🎯 AgentOrchestrator"]
        COMPILED["🔀 CompiledStateGraph"]
    end

    subgraph AGENTS["Agent Pipeline"]
        R["🔍 Researcher\n→ RAGFacts"]
        P["📅 Planner\n→ PlannerFacts"]
        O["⚖️ Operator\n→ OperatorFacts"]
        W["✍️ Writer\n→ final_response"]
        HITL["👨‍💼 Human Review\nHITL"]
    end

    subgraph TOOLS["External Services"]
        VDB2[("🗄️ Vector Store\nChroma / Supabase")]
        OSM["🗺️ OpenStreetMap\nOSRM"]
        CALC["🧮 Calculator\n+ FX API"]
    end

    subgraph PLATFORM["Platform"]
        LF2["📊 LangFuse\nTracing"]
        CP2[("💾 SQLite /\nPostgreSQL")]
    end

    FE -->|"SSE / REST"| API2
    CLI2 --> ORCH
    API2 --> ORCH
    ORCH --> COMPILED

    COMPILED --> R
    COMPILED --> P
    COMPILED --> O
    COMPILED --> W
    COMPILED -->|"interrupt_before"| HITL

    R -->|"RAG query"| VDB2
    P -->|"geo_tools"| OSM
    O -->|"safe_calculator"| CALC

    ORCH -->|"traces"| LF2
    COMPILED -->|"checkpoints"| CP2

    classDef uiStyle fill:#fce7f3,stroke:#ec4899,color:#831843
    classDef backendStyle fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef agentStyle fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef toolStyle fill:#fef3c7,stroke:#f59e0b,color:#78350f
    classDef platformStyle fill:#f1f5f9,stroke:#94a3b8,color:#1e293b

    class FE,CLI2 uiStyle
    class API2,ORCH,COMPILED backendStyle
    class R,P,O,W,HITL agentStyle
    class VDB2,OSM,CALC toolStyle
    class LF2,CP2 platformStyle
```

## Technology stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM abstraction | [LangChain](https://python.langchain.com/) |
| LLM providers | OpenAI · Google Gemini |
| Vector stores | ChromaDB (local) · Supabase pgvector (cloud) |
| REST API | [FastAPI](https://fastapi.tiangolo.com/) |
| Frontend | [Streamlit](https://streamlit.io/) |
| Config | [Dynaconf](https://www.dynaconf.com/) |
| Observability | [LangFuse](https://langfuse.com/) |
| Testing | [pytest](https://pytest.org/) |
