"""Regression tests for the confirmed findings of the v0.2.0 adversarial
review — each of these was a real, reproduced defect; the test names say
what used to go wrong."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wattage.ci import EXIT_CONFIG_ERROR, run_ci
from wattage.config import load_config
from wattage.models import LLMCall, RawSpan, SpanKind, TokenUsage
from wattage.normalize import normalize_llm_call
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import PricingConfigError, PricingRegistry


def test_google_and_xai_cache_tokens_are_inclusive_not_double_billed() -> None:
    """Gemini's promptTokenCount and Grok's prompt_tokens both INCLUDE
    cached tokens (their own API docs say so). Treating them as additive
    double-billed the cached tokens and pushed tier selection over the
    200k threshold — a ~5x silent overcharge in the reproduced case."""
    span = RawSpan(
        span_id="g1",
        trace_id="t",
        name="chat gemini-2.5-pro",
        kind=SpanKind.chat,
        attributes={
            "gen_ai.provider.name": "google",
            "gen_ai.request.model": "gemini-2.5-pro",
            "gen_ai.usage.input_tokens": 150_000,
            "gen_ai.usage.cache_read.input_tokens": 100_000,
            "gen_ai.usage.output_tokens": 1_000,
        },
    )
    call = normalize_llm_call(span)
    assert call.usage.input == 50_000  # 150k total prompt minus 100k cached
    assert call.usage.cache_read == 100_000

    cost = PricingEngine(PricingRegistry.load()).price_call(call)
    # Real prompt is 150k — BELOW the 200k tier, so the low-tier rates apply.
    assert cost.input == pytest.approx(50_000 * 1.25e-6)
    assert cost.cache_read == pytest.approx(100_000 * 0.125e-6)
    assert cost.total == pytest.approx(50_000 * 1.25e-6 + 100_000 * 0.125e-6 + 1_000 * 10.0e-6)


def test_prefix_churn_recoverable_dollars_use_absolute_cached_rates() -> None:
    """OpenAI/Google/xAI entries publish absolute cached-input rates and no
    cache_read_mult; computing recoverable savings from the multiplier
    (which defaults to 1.0) silently zeroed every prefix-churn dollar on
    34 of the 52 vendored models."""
    from wattage.config import WattageConfig
    from wattage.detectors.base import AnalysisContext
    from wattage.detectors.prefix_churn import PrefixChurnDetector
    from wattage.models import Session, Task

    engine = PricingEngine(PricingRegistry.load())

    def call(span_id: str, input_tok: int, output_tok: int, start: int) -> LLMCall:
        c = LLMCall(
            span_id=span_id,
            provider="openai",
            model="gpt-5.4",
            usage=TokenUsage(input=input_tok, output=output_tok),
            start_ns=start,
        )
        c.cost = engine.price_call(c)
        return c

    task = Task(
        task_id="t0",
        llm_calls=[call("a", 50_000, 2_000, 1), call("b", 60_000, 500, 2)],
    )
    session = Session(session_id="s0", tasks=[task])
    ctx = AnalysisContext(pricing=engine, config=WattageConfig())
    findings = PrefixChurnDetector().analyze(session, ctx)
    assert len(findings) == 1
    # 52k resent tokens at (input $2.5/M - cached $0.25/M) = $2.25/M.
    assert findings[0].wasted_dollars == pytest.approx(52_000 * (2.5e-6 - 0.25e-6))
    assert findings[0].wasted_dollars > 0


def test_openai_dashed_dated_sku_stays_unpriced_not_misresolved() -> None:
    """gpt-4o-2024-05-13 is a genuinely different price point ($5/$15, no
    caching) from gpt-4o ($2.50/$10) — resolving it to the base entry
    silently billed roughly half the real rate. Dashed dated SKUs must stay
    unpriced until the user supplies an override."""
    engine = PricingEngine(PricingRegistry.load())
    cost = engine.price_call(
        LLMCall(
            span_id="s",
            provider="openai",
            model="gpt-4o-2024-05-13",
            usage=TokenUsage(input=1_000_000, output=100_000),
        )
    )
    assert cost.unpriced is True
    # The compact Anthropic form still resolves (dateless id IS the alias).
    resolved = engine.price_call(
        LLMCall(
            span_id="s2",
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            usage=TokenUsage(input=1_000),
        )
    )
    assert resolved.unpriced is False


def test_explicit_null_rate_field_is_a_clean_config_error(tmp_path: Path) -> None:
    """`cache_read_mult: null` used to pass validation and then crash the
    engine with float(None) — a raw TypeError at exit 1, masquerading as a
    cost regression."""
    override = tmp_path / "o.yaml"
    override.write_text(
        "providers:\n  anthropic:\n    claude-sonnet-5:\n      cache_read_mult: null\n"
    )
    with pytest.raises(PricingConfigError, match="null"):
        PricingRegistry.load(overrides_path=str(override))


def test_scalar_override_on_dated_rates_model_is_rejected_not_ignored(tmp_path: Path) -> None:
    """A top-level `input:` override on a model priced through dated rate
    windows used to load cleanly and then do nothing — the windows' values
    always won. Now it's a config error with the fix in the message."""
    override = tmp_path / "o.yaml"
    override.write_text("providers:\n  google:\n    gemini-3.7-flash:\n      input: 9.99e-6\n")
    with pytest.raises(PricingConfigError, match="rates"):
        PricingRegistry.load(overrides_path=str(override))


def test_quality_file_json_array_is_config_error_not_typeerror(tmp_path: Path) -> None:
    trace = tmp_path / "t.json"
    trace.write_text(
        json.dumps(
            {
                "resourceSpans": [
                    {
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "traceId": "t1",
                                        "spanId": "s1",
                                        "name": "chat",
                                        "attributes": [
                                            {
                                                "key": "gen_ai.provider.name",
                                                "value": {"stringValue": "anthropic"},
                                            },
                                            {
                                                "key": "gen_ai.request.model",
                                                "value": {"stringValue": "claude-sonnet-5"},
                                            },
                                            {
                                                "key": "gen_ai.usage.input_tokens",
                                                "value": {"intValue": "100"},
                                            },
                                            {
                                                "key": "gen_ai.usage.output_tokens",
                                                "value": {"intValue": "10"},
                                            },
                                        ],
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        )
    )
    bad_quality = tmp_path / "q.json"
    bad_quality.write_text("[1, 2]")
    result = run_ci(
        str(trace),
        baseline_path=str(tmp_path / "b.json"),
        quality_file=str(bad_quality),
    )
    assert result.exit_code == EXIT_CONFIG_ERROR


def test_judge_off_unquoted_yaml_boolean_loads(tmp_path: Path) -> None:
    """YAML parses unquoted `off` as boolean False — and `judge: off` is the
    exact spelling the docs recommend. It must load as the string "off"."""
    cfg_file = tmp_path / "wattage.yaml"
    cfg_file.write_text("detectors:\n  nonconvergence:\n    judge: off\n")
    cfg = load_config(str(cfg_file))
    assert cfg.detectors.nonconvergence.judge == "off"


def test_ci_verdict_survives_bracketed_trace_text() -> None:
    """Failure reasons embed trace-derived text; a bracketed value used to
    raise rich MarkupError and crash with Python's default exit 1 — a fake
    cost regression."""
    import io

    from rich.console import Console

    from wattage.ci import CIResult
    from wattage.render.terminal import render_ci_verdict

    buf = io.StringIO()
    console = Console(file=buf, width=100, no_color=True, force_terminal=False)
    render_ci_verdict(
        CIResult(exit_code=3, reasons=["adapter error near text [/bold] from trace"]),
        baseline_path="x",
        baseline_updated=True,
        console=console,
    )
    assert "[/bold]" in buf.getvalue()


def test_report_survives_bracketed_tool_and_model_names() -> None:
    """Trace-controlled names rendered into rich markup used to crash the
    report (or silently restyle it)."""
    import io

    from rich.console import Console

    from wattage.models import Finding, FindingBasis, QualityRisk, Report, Score, Severity
    from wattage.render.terminal import render_terminal

    report = Report(
        trace_source="t.json",
        total_dollars=1.0,
        token_breakdown={
            "input": 10,
            "output": 5,
            "cache_read": 0,
            "cache_creation": 0,
            "reasoning": 0,
        },
        findings=[
            Finding(
                id="tool_result_bloat",
                severity=Severity.high,
                wasted_dollars=0.1,
                basis=FindingBasis.estimated,
                quality_risk=QualityRisk.low,
                evidence="oversized results",
                fix="Truncate '[/bold]evil_tool' output at the tool boundary.",
                span_ids=["a"],
            )
        ],
        score=Score(
            efficiency=90,
            grade="A",
            waste_ratio=0.0,
            quality_factor=1.0,
            quality_measured=False,
            recoverable_dollars=0.0,
        ),
        pricing_version="test",
        generated_at="2026-08-23T00:00:00+00:00",
        unpriced_calls=1,
        unpriced_models=["openai/[/bold]evil-model"],
    )
    buf = io.StringIO()
    console = Console(file=buf, width=100, no_color=True, force_terminal=False)
    render_terminal(report, console=console)
    out = buf.getvalue()
    assert "evil_tool" in out and "evil-model" in out
