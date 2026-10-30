"""Deterministic, versioned safety rule engine. The LLM never overrides its output."""
from . import data


def _match(cond: dict, state: dict) -> bool:
    if "any" in cond:
        return any(_match(c, state) for c in cond["any"])
    if "all" in cond:
        return all(_match(c, state) for c in cond["all"])
    value = state.get(cond["field"])
    if "in" in cond:
        return value in cond["in"]
    if "has_any" in cond:
        return bool(set(value or []) & set(cond["has_any"]))
    raise ValueError(f"Unknown condition: {cond}")


def evaluate(state: dict) -> list[dict]:
    """Return fired rules as [{id, level}], most urgent first."""
    fired = [
        {"id": r["id"], "level": r["level"]}
        for r in data.rules()["rules"]
        if _match(r["when"], state)
    ]
    return sorted(fired, key=lambda r: -data.LEVEL_ORDER[r["level"]])


def version() -> str:
    return data.rules()["version"]
