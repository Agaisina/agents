# Adding a New Agent

Adding a new agent to the system requires **three steps** and zero changes to the graph or orchestrator.

## 1. Write the YAML prompt

Create `src/agents/application/prompts/<role_name>.yaml`:

```yaml
agent:
  name: "Senior Budget Analyst"
  temperature: 0.05
  prompt: |
    You are the Senior Budget Analyst for Veritas Corporate.
    Your job is to …
```

## 2. Register the definition

Open `src/agents/application/agent_definitions.py` and add an entry to `AGENT_DEFINITIONS`:

```python
AGENT_DEFINITIONS: Dict[str, AgentDefinition] = {
    # … existing agents …
    "budget_analyst": AgentDefinition(
        prompt_key="budget_analyst",
        default_tool_keys=["safe_calculator"],   # optional
    ),
}
```

If the agent needs a brand-new tool, add it to `TOOL_REGISTRY` in the same file.

## 3. Wire the agent into the graph

In `src/workflow/graph.py`, inside `WorkflowBuilder._setup_agents()`:

```python
self.budget_analyst_agent = self.registry.get_agent("budget_analyst")
```

Then add a new node and edge in `compile()`:

```python
workflow.add_node("budget_analyst", self._budget_analyst_node)
workflow.add_edge(OPERATOR_EXTRACTOR_NODE, "budget_analyst")
workflow.add_edge("budget_analyst", WRITER_NODE)
```

That's it — the factory and registry handle LLM instantiation automatically.

!!! tip
    Run `make test` after adding an agent to verify the registry loads it correctly.
