# Agent Pipeline

## Node sequence

```mermaid
flowchart TD
    START(["▶ START"])
    END_NODE(["⏹ END"])

    R["🔍 Researcher\nExtracts RAGFacts from policy knowledge base"]
    P["📅 Planner\nBuilds itinerary · calls geo & transport tools"]
    PT(["🛠️ Planner Tools\ngeo_tools · transport APIs"])
    PE["📊 Planner Extractor\nStructures PlannerFacts"]
    O["⚖️ Operator\nAudits plan · calculates TCO"]
    OT(["🧮 Operator Tools\nsafe_calculator · convert_currency"])
    OE["📑 Operator Extractor\nStructures OperatorFacts"]
    W["✍️ Writer\nGenerates employee-facing report"]
    HR["👨‍💼 Human Review\nManager approval — HITL interrupt"]

    START --> R
    R --> P

    P -->|"tool_calls?"| PT
    PT -->|"tool results"| P
    P -->|"done"| PE

    PE --> O

    O -->|"tool_calls?"| OT
    OT -->|"tool results"| O
    O -->|"done"| OE

    OE -->|"AUTO_APPROVED ✅"| W
    OE -->|"NEEDS_REVIEW ⏸️"| HR
    OE -->|"REJECTED_MUST_REPLAN ❌"| P

    HR -->|"APPROVED ✅"| W
    HR -->|"REJECTED ❌"| P

    W --> END_NODE

    classDef startEnd fill:#f1f5f9,stroke:#64748b,color:#1e293b
    classDef agentNode fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef toolNode fill:#ede9fe,stroke:#8b5cf6,color:#3b0764
    classDef extractNode fill:#d1fae5,stroke:#10b981,color:#064e3b
    classDef writerNode fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef humanNode fill:#fee2e2,stroke:#ef4444,color:#7f1d1d

    class START,END_NODE startEnd
    class R,P,O agentNode
    class PT,OT toolNode
    class PE,OE extractNode
    class W writerNode
    class HR humanNode
```

## Approval routing

The `OperatorFacts.approval_routing` field controls post-operator routing:

```mermaid
flowchart LR
    OE["Operator Extractor"]

    OE -->|"AUTO_APPROVED"| A["✅ Writer\n100% compliant"]
    OE -->|"REJECTED_MUST_REPLAN"| B["❌ Planner\nHard violation"]
    OE -->|"PENDING_MANAGER"| C["⏸️ Human Review\nMinor overage &lt; 20%"]
    OE -->|"PENDING_VP_FINANCE"| D["⏸️ Human Review\nMajor overage &gt; 20%"]
    OE -->|"PENDING_CEO_SECURITY"| E["⏸️ Human Review\nTier-3 risk / Tier-1 client"]
    OE -->|"INCOMPLETE_DATA"| F["⏸️ Human Review\nMissing pricing data"]

    classDef ok fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef ko fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    classDef pending fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef oe fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f

    class OE oe
    class A ok
    class B ko
    class C,D,E,F pending
```

## Human-in-the-Loop (HITL)

When routing reaches **Human Review**, LangGraph suspends execution via `interrupt_before`. The graph state is persisted to the checkpointer. The Manager UI calls `POST /v1/chat/resume` with `approve: true/false` and an optional `feedback` message. The `AgentOrchestrator.aresume_suspended_query()` method injects the decision and resumes execution.
