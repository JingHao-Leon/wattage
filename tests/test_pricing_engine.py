from datetime import datetime, timezone
from pathlib import Path

import pytest

from wattage.models import LLMCall, TokenUsage
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import PricingConfigError, PricingRegistry


@pytest.fixture
def engine() -> PricingEngine:
    return PricingEngine(PricingRegistry.load())


def test_prices_known_model_across_all_token_classes(engine: PricingEngine) -> None:
    call = LLMCall(
        span_id="s1",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(input=10_000, output=300, cache_read=40_000, cache_creation=2_000),
    )
    cost = engine.price_call(call)

    assert cost.unpriced is False
    assert cost.input == pytest.approx(10_000 * 3.0e-6)
    assert cost.output == pytest.approx(300 * 15.0e-6)
    assert cost.cache_read == pytest.approx(40_000 * 3.0e-6 * 0.10)
    assert cost.cache_creation == pytest.approx(2_000 * 3.0e-6 * 1.25)
    assert cost.total == pytest.approx(
        cost.input + cost.output + cost.cache_read + cost.cache_creation
    )
    assert cost.pricing_version == "2026-08-23-verified"


def test_reasoning_tokens_price_at_the_output_rate(engine: PricingEngine) -> None:
    call = LLMCall(
        span_id="s1r",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(input=100, output=50, reasoning=400),
    )
    cost = engine.price_call(call)
    assert cost.reasoning == pytest.approx(400 * 15.0e-6)
    assert cost.output == pytest.approx(50 * 15.0e-6)
    assert cost.total == pytest.approx(cost.input + cost.output + cost.reasoning)


def test_unknown_model_is_left_unpriced_without_a_raw_warning(engine: PricingEngine) -> None:
    """The unpriced signal travels on Cost.unpriced (surfaced by the Report
    and the renderers), not a raw Python UserWarning leaking site-packages
    paths into the terminal output."""
    import warnings

    call = LLMCall(
        span_id="s2", provider="mystery-provider", model="ghost-model", usage=TokenUsage(input=100)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cost = engine.price_call(call)
    assert caught == []
    assert cost.unpriced is True
    assert cost.total == 0
    assert cost.input == 0


def test_dated_model_id_resolves_to_its_base_entry(engine: PricingEngine) -> None:
    """Real traces carry pinned ids like claude-sonnet-4-5-20250929; the
    registry resolves them to the base entry and records the provenance."""
    call = LLMCall(
        span_id="s3",
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        usage=TokenUsage(input=1_000),
    )
    cost = engine.price_call(call)
    assert cost.unpriced is False
    assert cost.input == pytest.approx(1_000 * 3.0e-6)
    assert cost.priced_as == "claude-sonnet-4-5"


def test_absolute_cache_read_rate_wins_over_multiplier(engine: PricingEngine) -> None:
    """gpt-4o's cached-input rate is $1.25/MTok — 0.5x input, not the 0.1x a
    shared multiplier would guess. The absolute per-token rate is what's
    billed."""
    call = LLMCall(
        span_id="s4",
        provider="openai",
        model="gpt-4o",
        usage=TokenUsage(input=1_000, cache_read=10_000),
    )
    cost = engine.price_call(call)
    assert cost.cache_read == pytest.approx(10_000 * 1.25e-6)


def test_context_tier_reprices_the_whole_request(engine: PricingEngine) -> None:
    """xAI reprices ALL tokens once the prompt reaches 200k tokens; below
    the threshold the base rate applies."""
    small = LLMCall(
        span_id="s5a",
        provider="xai",
        model="grok-4.6",
        usage=TokenUsage(input=100_000, output=1_000),
    )
    large = LLMCall(
        span_id="s5b",
        provider="xai",
        model="grok-4.6",
        usage=TokenUsage(input=250_000, output=1_000),
    )
    small_cost = engine.price_call(small)
    large_cost = engine.price_call(large)
    assert small_cost.input == pytest.approx(100_000 * 2.0e-6)
    assert small_cost.output == pytest.approx(1_000 * 6.0e-6)
    assert large_cost.input == pytest.approx(250_000 * 4.0e-6)
    assert large_cost.output == pytest.approx(1_000 * 12.0e-6)


def test_published_but_rateless_tier_is_unpriced_not_guessed(engine: PricingEngine) -> None:
    """OpenAI's pricing page shows gpt-5.5 rates only for <272K context; a
    larger prompt has no published rate and must be unpriced, never billed
    at the wrong tier."""
    call = LLMCall(
        span_id="s6",
        provider="openai",
        model="gpt-5.5",
        usage=TokenUsage(input=300_000, output=100),
    )
    cost = engine.price_call(call)
    assert cost.unpriced is True
    assert cost.total == 0


def test_dated_rate_window_prices_by_call_timestamp(engine: PricingEngine) -> None:
    """Gemini 3.7 Flash promo pricing runs through 2026-12-31; a call
    timestamped in 2027 prices at the post-promo rate."""
    promo_ns = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    later_ns = int(datetime(2027, 2, 1, tzinfo=timezone.utc).timestamp() * 1e9)
    promo_call = LLMCall(
        span_id="s7a",
        provider="google",
        model="gemini-3.7-flash",
        usage=TokenUsage(input=1_000_000),
        start_ns=promo_ns,
        end_ns=promo_ns,
    )
    later_call = LLMCall(
        span_id="s7b",
        provider="google",
        model="gemini-3.7-flash",
        usage=TokenUsage(input=1_000_000),
        start_ns=later_ns,
        end_ns=later_ns,
    )
    assert engine.price_call(promo_call).input == pytest.approx(1_000_000 * 0.75e-6)
    assert engine.price_call(later_call).input == pytest.approx(1_000_000 * 1.5e-6)


def test_cache_write_ttl_breakdown_prices_1h_writes_at_their_real_rate(
    engine: PricingEngine,
) -> None:
    """Claude Code sessions carry the 5m/1h cache-write split; 1-hour writes
    bill at 2x base input instead of 1.25x. Without the breakdown, every
    write prices at the default 5-minute rate."""
    with_breakdown = LLMCall(
        span_id="s8a",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(input=100, cache_creation=10_000),
        cache_write_ttl_breakdown={"5m": 4_000, "1h": 6_000},
    )
    without_breakdown = LLMCall(
        span_id="s8b",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(input=100, cache_creation=10_000),
    )
    with_cost = engine.price_call(with_breakdown)
    without_cost = engine.price_call(without_breakdown)
    assert with_cost.cache_creation == pytest.approx(4_000 * 3.0e-6 * 1.25 + 6_000 * 3.0e-6 * 2.0)
    assert without_cost.cache_creation == pytest.approx(10_000 * 3.0e-6 * 1.25)


def test_mismatched_ttl_breakdown_is_ignored_not_trusted(engine: PricingEngine) -> None:
    """A breakdown that doesn't sum to cache_creation is inconsistent data —
    the safe default (5-minute rate on the total) applies instead."""
    call = LLMCall(
        span_id="s9",
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(input=100, cache_creation=10_000),
        cache_write_ttl_breakdown={"5m": 1, "1h": 2},
    )
    cost = engine.price_call(call)
    assert cost.cache_creation == pytest.approx(10_000 * 3.0e-6 * 1.25)


def test_registry_overrides_merge_over_vendored(tmp_path: Path) -> None:
    override_path = tmp_path / "overrides.yaml"
    override_path.write_text(
        "version: 'test-override'\n"
        "providers:\n"
        "  anthropic:\n"
        "    claude-sonnet-4-6:\n"
        "      input: 1.0e-6\n"
        "      output: 2.0e-6\n"
        "      cache_read_mult: 0.5\n"
        "      cache_write_mult: 1.0\n"
        "      min_cacheable_prefix_tokens: 1024\n"
    )
    registry = PricingRegistry.load(overrides_path=str(override_path))
    assert registry.version == "test-override"
    price = registry.get("anthropic", "claude-sonnet-4-6")
    assert price.input == 1.0e-6
    assert price.cache_read_mult == 0.5

    # An untouched model from the vendored snapshot must still resolve.
    haiku_price = registry.get("anthropic", "claude-haiku-4-5")
    assert haiku_price.input == 1.0e-6


def test_partial_override_keeps_vendored_cache_multipliers(tmp_path: Path) -> None:
    """Overriding just input/output must NOT silently reset the vendored
    cache multipliers to 1.0 (a 10x cache-read overcharge) — overrides merge
    field-level, not whole-entry."""
    override_path = tmp_path / "overrides.yaml"
    override_path.write_text(
        "providers:\n  anthropic:\n    claude-sonnet-4-6:\n      input: 9.0e-6\n"
    )
    registry = PricingRegistry.load(overrides_path=str(override_path))
    price = registry.get("anthropic", "claude-sonnet-4-6")
    assert price.input == 9.0e-6
    assert price.output == 15.0e-6  # vendored value preserved
    assert price.cache_read_mult == 0.10  # NOT reset to 1.0
    assert price.min_cacheable_prefix_tokens == 1024


def test_string_typed_price_is_a_clean_config_error(tmp_path: Path) -> None:
    """YAML parses a quoted '3.0e-6' as a string; the registry must reject
    it as a config error (exit 2), not die later in the engine with a
    string-multiplication TypeError."""
    override_path = tmp_path / "overrides.yaml"
    override_path.write_text(
        'providers:\n  anthropic:\n    claude-sonnet-4-6:\n      input: "3.0e-6"\n'
    )
    with pytest.raises(PricingConfigError, match="must be a number"):
        PricingRegistry.load(overrides_path=str(override_path))


def test_null_provider_body_is_a_clean_config_error(tmp_path: Path) -> None:
    override_path = tmp_path / "overrides.yaml"
    override_path.write_text("providers:\n  anthropic:\n")
    with pytest.raises(PricingConfigError, match="anthropic"):
        PricingRegistry.load(overrides_path=str(override_path))
