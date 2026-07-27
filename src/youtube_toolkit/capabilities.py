"""Optional dependency capabilities and their user-facing installation guidance."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

Capability = Literal["storage", "vision", "llm"]

CAPABILITY_MODULES: dict[Capability, tuple[str, ...]] = {
    "storage": ("lancedb", "platformdirs", "tomlkit"),
    "vision": ("numpy", "PIL"),
    "llm": ("langchain_core", "langchain_openai", "pydantic"),
}


def _dependency_available(module: str) -> bool:
    """Return whether one dependency can be imported without importing it."""
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def capability_available(capability: Capability) -> bool:
    """Return whether every dependency owned by one optional capability is installed."""
    return all(_dependency_available(module) for module in CAPABILITY_MODULES[capability])


def require_capability(capability: Capability, feature: str) -> None:
    """Reject an unavailable optional feature with capability-level installation guidance."""
    if capability_available(capability):
        return
    raise RuntimeError(f"{feature} requires the {capability} capability. Install youtube-toolkit[{capability}].")
