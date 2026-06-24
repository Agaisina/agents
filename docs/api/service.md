# Orchestrator (Service)

`AgentOrchestrator` is the single entry point for all callers (REST API, CLI, tests).
It owns the compiled graph, the agent registry, the vector store, and the checkpointer.

## AgentOrchestrator

::: src.workflow.service.AgentOrchestrator
