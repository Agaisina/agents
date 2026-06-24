import functools
import yaml
from pathlib import Path
from typing import Any, Dict


@functools.lru_cache(maxsize=None)
def load_agent_prompt(agent_key: str) -> Dict[str, Any]:
    """Load and cache a YAML prompt file for a given agent key.

    The file is resolved relative to the ``prompts/`` directory next to this
    module. Results are cached indefinitely so repeated calls within the same
    process pay no I/O cost.

    Args:
        agent_key: The filename stem, e.g. ``"planner"`` resolves to
                   ``prompts/planner.yaml``.

    Returns:
        The parsed YAML content as a plain dictionary.

    Raises:
        FileNotFoundError: When no YAML file exists for ``agent_key``.
    """
    base_path = Path(__file__).parent / "prompts"
    file_path = base_path / f"{agent_key}.yaml"

    if not file_path.exists():
        raise FileNotFoundError(f"Config for agent '{agent_key}' not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)