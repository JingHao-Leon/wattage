"""Detector ABC + AnalysisContext (doc §9.2), plus shared call-traversal
helpers every detector needs (a task's LLM/tool calls, in chronological
order, across both direct task calls and loop iterations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import entry_points
from typing import Any

from wattage.config import WattageConfig
from wattage.models import Finding, LLMCall, Session, Task, ToolCall
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import ModelPrice, PricingRegistry


@dataclass
class AnalysisContext:
    pricing: PricingEngine
    config: WattageConfig
    embedder: Any | None = None
    judge: Any | None = None
    quality_map: dict[str, Any] | None = None


class Detector(ABC):
    id: str
    default_enabled: bool = True

    @abstractmethod
    def analyze(self, session: Session, ctx: AnalysisContext) -> list[Finding]:
        """Pure function of a session (+ ctx). Deterministic, side-effect free."""


def call_timestamp(call: LLMCall) -> datetime | None:
    """The call's own wall-clock time, for dated rate windows."""
    ns = call.end_ns or call.start_ns
    if ns <= 0:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def rate_card_for(registry: PricingRegistry, call: LLMCall) -> ModelPrice:
    """Registry lookup pinned to the call's own timestamp, so detectors
    price waste at the rate window the call was actually billed under — not
    whatever window happens to cover the day the analysis runs (a trace
    from a promo-pricing period must not have its waste priced at the
    post-promo rate). Raises UnknownModelError like registry.get."""
    return registry.get(call.provider, call.model, at=call_timestamp(call))


def ordered_llm_calls(task: Task) -> list[LLMCall]:
    calls = list(task.llm_calls)
    for loop in task.loops:
        for iteration in loop.iterations:
            calls.extend(iteration.llm_calls)
    return sorted(calls, key=lambda c: c.start_ns)


def ordered_tool_calls(task: Task) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for loop in task.loops:
        for iteration in loop.iterations:
            calls.extend(iteration.tool_calls)
    return sorted(calls, key=lambda c: c.start_ns)


def load_detectors(config: WattageConfig) -> list[Detector]:
    detectors: list[Detector] = []
    for ep in entry_points(group="wattage.detectors"):
        detector_cls = ep.load()
        instance: Detector = detector_cls()
        detector_cfg = getattr(config.detectors, instance.id, None)
        enabled = detector_cfg.enabled if detector_cfg is not None else instance.default_enabled
        if enabled:
            detectors.append(instance)
    return detectors
