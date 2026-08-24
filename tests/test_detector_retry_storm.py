"""retry_storm detector tests: consecutive same-model calls with
near-identical prompt-side token counts are the shape of a client retry
loop re-billing the full prompt every attempt.

The dollars reported are the real billed cost of the repeat attempts, but
whether they were retries (vs a deliberate fixed-prompt loop) is a
heuristic call — so the basis must stay `estimated`, visible but never
able to fail a CI build on its own.
"""

import math

from hypothesis import given
from hypothesis import strategies as st

from wattage.config import WattageConfig
from wattage.detectors.base import AnalysisContext
from wattage.detectors.retry_storm import RetryStormDetector
from wattage.models import Finding, FindingBasis, LLMCall, Session, Severity, Task, TokenUsage
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import PricingRegistry


def _call(
    engine: PricingEngine,
    span_id: str,
    input_tokens: int,
    start: int,
    model: str = "claude-sonnet-4-6",
    cache_read: int = 0,
    cache_creation: int = 0,
    output_tokens: int = 50,
) -> LLMCall:
    call = LLMCall(
        span_id=span_id,
        provider="anthropic",
        model=model,
        usage=TokenUsage(
            input=input_tokens,
            output=output_tokens,
            cache_read=cache_read,
            cache_creation=cache_creation,
        ),
        start_ns=start,
    )
    call.cost = engine.price_call(call)
    return call


def _analyze(engine: PricingEngine, calls: list[LLMCall]) -> list[Finding]:
    session = Session(session_id="s0", tasks=[Task(task_id="t0", llm_calls=calls)])
    ctx = AnalysisContext(pricing=engine, config=WattageConfig())
    return RetryStormDetector().analyze(session, ctx)


def test_three_near_identical_calls_flag_the_repeats_real_cost() -> None:
    """Three back-to-back same-model calls within the 2% shape tolerance
    are one finding spanning all three, whose dollars are the actual billed
    cost of the 2nd+3rd attempts (the first attempt was going to happen
    anyway). Prompt shape is the SUM of input+cache_read+cache_creation, so
    a retry that re-sends the same context through the cache still matches."""
    engine = PricingEngine(PricingRegistry.load())
    calls = [
        _call(engine, "r1", input_tokens=600, cache_read=400, start=10),  # prompt-side 1000
        _call(engine, "r2", input_tokens=1010, start=20),
        _call(engine, "r3", input_tokens=1005, start=30),
    ]

    findings = _analyze(engine, calls)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.span_ids == ["r1", "r2", "r3"]
    assert finding.basis == FindingBasis.estimated
    assert finding.severity == Severity.medium
    assert finding.wasted_dollars == calls[1].cost.total + calls[2].cost.total
    assert finding.wasted_tokens == calls[1].usage.total() + calls[2].usage.total()


def test_two_repeats_stay_below_min_repeats() -> None:
    """Two same-shaped calls in a row is normal agent behavior (an appended
    tool result can leave the prompt size nearly unchanged) — the default
    min_repeats of 3 must not fire on it."""
    engine = PricingEngine(PricingRegistry.load())
    calls = [
        _call(engine, "r1", input_tokens=1000, start=10),
        _call(engine, "r2", input_tokens=1000, start=20),
    ]
    assert _analyze(engine, calls) == []


def test_prompt_size_change_beyond_tolerance_breaks_the_run() -> None:
    engine = PricingEngine(PricingRegistry.load())
    calls = [
        _call(engine, "r1", input_tokens=1000, start=10),
        _call(engine, "r2", input_tokens=1005, start=20),
        _call(engine, "r3", input_tokens=1200, start=30),  # ~16% jump: context grew
    ]
    assert _analyze(engine, calls) == []


def test_model_change_breaks_the_run() -> None:
    """A different model mid-sequence is a routing decision, not a retry —
    even with identical prompt sizes the run must reset."""
    engine = PricingEngine(PricingRegistry.load())
    calls = [
        _call(engine, "r1", input_tokens=1000, start=10),
        _call(engine, "r2", input_tokens=1000, start=20),
        _call(engine, "r3", input_tokens=1000, start=30, model="claude-haiku-4-5"),
    ]
    assert _analyze(engine, calls) == []


def test_five_repeats_escalate_to_high_severity() -> None:
    engine = PricingEngine(PricingRegistry.load())
    calls = [_call(engine, f"r{i}", input_tokens=1000, start=i * 10) for i in range(5)]

    findings = _analyze(engine, calls)

    assert len(findings) == 1
    assert findings[0].severity == Severity.high
    assert findings[0].span_ids == [f"r{i}" for i in range(5)]


@given(base=st.integers(min_value=1, max_value=2000), steps=st.integers(min_value=3, max_value=8))
def test_property_strictly_growing_prompts_are_never_flagged(base: int, steps: int) -> None:
    """A healthy agent loop accumulates context, so prompt sizes grow every
    turn. Any sequence where each prompt is at least 1.2x the previous (a
    ~17%+ step, far outside the 2% tolerance) must never read as a retry
    storm, whatever the base size or run length."""
    sizes = [base]
    for _ in range(steps - 1):
        sizes.append(math.ceil(sizes[-1] * 1.2))
    engine = PricingEngine(PricingRegistry.load())
    calls = [
        _call(engine, f"g{i}", input_tokens=size, start=i * 10) for i, size in enumerate(sizes)
    ]
    assert _analyze(engine, calls) == []
