# State & Checkpointing

## AgentState

The `AgentState` TypedDict is the single shared mutable object that flows through all graph nodes.

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rag_facts: Optional[RAGFacts]
    planner_facts: Optional[PlannerFacts]
    operator_facts: Optional[OperatorFacts]
    manager_override: Optional[bool]
    final_response: Optional[str]
    planner_tool_iterations: int
    planner_turn_start: int
    operator_turn_start: int
```

The `messages` field uses the `add_messages` reducer — each node **appends** rather than replaces messages, so the full conversation history is always available.

`planner_turn_start` and `operator_turn_start` are index markers that let extractor nodes scope their context to just the current agent cycle, avoiding token waste.

## Checkpointing

| Environment | Backend | Setting |
|---|---|---|
| Local dev | SQLite (in-memory or file) | `database.type: sqlite` |
| Production | Supabase / PostgreSQL | `database.type: postgres` |

The `CheckpointFactory` reads `settings.database.type` at startup and returns the correct LangGraph saver. This decouples the graph from storage details.

Every `thread_id` identifies a unique conversation. Replaying or resuming a conversation after HITL interruption uses the same `thread_id` — LangGraph reconstructs state from the checkpoint automatically.
