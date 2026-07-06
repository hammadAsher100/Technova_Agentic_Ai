"""Generic tool-routing scaffold (Section 10).

Zero tools are registered in Phase 0 — this file only provides the
registry mechanism. Phase 1 registers task-specific tools here IF the
uploaded task needs capabilities beyond LLM calls (Section 12, Phase 1
item 6). If it doesn't, this file can stay exactly as-is.
"""
import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Any]


class ToolRouter:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self._tools[name] = fn
        logger.info("tool_router.registered", extra={"tool": name})

    def route(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"No tool registered under name '{name}'")
        return self._tools[name](**kwargs)

    @property
    def available_tools(self) -> Dict[str, ToolFn]:
        return dict(self._tools)
