"""ingest -> sessionize -> price -> detect -> score -> Report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from wattage.adapters import resolve_adapter
from wattage.adapters.base import AdapterError
from wattage.config import WattageConfig
from wattage.convergence.embed import build_embedder
from wattage.detectors.base import AnalysisContext, load_detectors, ordered_llm_calls
from wattage.models import Report, Trace
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import PricingRegistry
from wattage.scoring.quality import QualityMapError, compute_quality_factor
from wattage.scoring.score import compute_score
from wattage.sessionize import sessionize


def load_quality_map(quality_file: str | None) -> dict[str, Any] | None:
    if quality_file is None:
        return None
    with open(quality_file, encoding="utf-8") as f:
        loaded = json.load(f)
    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        # A JSON array here would crash dict() with a TypeError deep in the
        # pipeline at the wrong exit code — this is a config-shaped input,
        # so it gets the config error type.
        raise QualityMapError(
            f"{quality_file!r} must contain a JSON object, got {type(loaded).__name__}"
        )
    return dict(loaded) if loaded else None


def build_report(
    source: str,
    pricing_override: str | None = None,
    config: WattageConfig | None = None,
    quality_file: str | None = None,
    runs_per_month: int | None = None,
) -> Report:
    _, report = build_trace_and_report(
        source,
        pricing_override=pricing_override,
        config=config,
        quality_file=quality_file,
        runs_per_month=runs_per_month,
    )
    return report


def build_trace_and_report(
    source: str,
    pricing_override: str | None = None,
    config: WattageConfig | None = None,
    quality_file: str | None = None,
    registry: PricingRegistry | None = None,
    quality_map: dict[str, Any] | None = None,
    runs_per_month: int | None = None,
) -> tuple[Trace, Report]:
    """Same pipeline as build_report, but also returns the full session/task/
    loop/iteration hierarchy — needed by the HTML flame graph renderer, which
    a flattened Report can't provide.

    `registry`/`quality_map` allow a caller (`wattage ci`) to load config
    inputs up front, so a malformed pricing/quality FILE fails as a config
    error before trace ingestion ever starts — the exit-code contract
    depends on that ordering.
    """
    config = config or WattageConfig()
    if quality_map is None:
        quality_map = load_quality_map(quality_file)
    if registry is None:
        registry = PricingRegistry.load(overrides_path=pricing_override)

    adapter = resolve_adapter(source)
    raw_spans = list(adapter.read(source))
    trace = sessionize(raw_spans, source=source)
    if not trace.sessions:
        # An empty trace produced zero real data to grade -- a confident
        # "A (100) / this trace looks efficient" here would be exactly the
        # plausible-looking-but-vacuous number this project exists to
        # avoid. `wattage ci` already treated this as an ingestion error;
        # raising it here means every caller (report/score/badge/ci) gets
        # the same treatment from one place instead of duplicating the
        # check.
        raise AdapterError(f"'{source}' produced zero sessions (empty or unparseable trace)")

    engine = PricingEngine(registry)

    token_breakdown = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
        "reasoning": 0,
    }
    total_dollars = 0.0
    unpriced_calls = 0
    unpriced_models: set[str] = set()

    # Price every call first: detectors (cache_gap, prefix_churn's ratio-based
    # severity) read call.cost, so pricing must happen before detection runs.
    for session in trace.sessions:
        for task in session.tasks:
            for call in ordered_llm_calls(task):
                call.cost = engine.price_call(call)
                token_breakdown["input"] += call.usage.input
                token_breakdown["output"] += call.usage.output
                token_breakdown["cache_read"] += call.usage.cache_read
                token_breakdown["cache_creation"] += call.usage.cache_creation
                token_breakdown["reasoning"] += call.usage.reasoning
                total_dollars += call.cost.total
                if call.cost.unpriced:
                    unpriced_calls += 1
                    unpriced_models.add(f"{call.provider}/{call.model}")

    if sum(token_breakdown.values()) == 0:
        # Spans exist but not one carried a token count: the classic
        # "instrumentation on, usage capture off" misconfiguration — or
        # every token-bearing span was quarantined as malformed. A
        # confident A (100) on zero data would be a vacuous pass — refuse
        # to grade, and say what's missing.
        if trace.skipped_spans:
            shown = "; ".join(trace.skipped_spans[:3])
            more = len(trace.skipped_spans) - 3
            suffix = f" (+{more} more)" if more > 0 else ""
            raise AdapterError(
                f"'{source}' has no valid usage data to grade — every token-bearing "
                f"span was quarantined as malformed: {shown}{suffix}"
            )
        raise AdapterError(
            f"'{source}' contains spans but zero token usage — no gen_ai.usage.* "
            "attributes were captured, so there is nothing real to grade. Enable "
            "usage/token capture in your instrumentation and re-export."
        )

    # Built once and shared: retrieval_thrash (Phase 2.10) reuses the same
    # embedder instance rather than each detector loading its own model.
    embedder = build_embedder(config.detectors.nonconvergence.embed)
    ctx = AnalysisContext(pricing=engine, config=config, embedder=embedder, quality_map=quality_map)
    detectors = load_detectors(config)
    findings = [
        finding
        for session in trace.sessions
        for detector in detectors
        for finding in detector.analyze(session, ctx)
    ]

    quality_factor, quality_measured = compute_quality_factor(
        quality_map, target=config.quality.target
    )
    score = compute_score(
        findings=findings,
        total_dollars=total_dollars,
        quality_factor=quality_factor,
        quality_measured=quality_measured,
        monthly_projection=None,
    )
    if runs_per_month is not None and runs_per_month > 0:
        # The only extrapolation offered is the one the user explicitly
        # asked for: recoverable dollars × their own stated run rate. Never
        # inferred from the trace's duration — a 3-minute demo trace says
        # nothing about production cadence.
        score.monthly_projection = score.recoverable_dollars * runs_per_month

    report = Report(
        trace_source=source,
        source_format=adapter.format_name,
        total_dollars=total_dollars,
        token_breakdown=token_breakdown,
        findings=findings,
        score=score,
        pricing_version=registry.version,
        generated_at=datetime.now(timezone.utc).isoformat(),
        unpriced_calls=unpriced_calls,
        unpriced_models=sorted(unpriced_models),
        dropped_token_spans=trace.dropped_token_spans,
        duplicate_spans=trace.duplicate_spans,
        skipped_spans=trace.skipped_spans,
    )
    return trace, report
