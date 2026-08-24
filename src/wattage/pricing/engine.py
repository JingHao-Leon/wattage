"""Cost Engine (doc §9.4): prices an LLMCall across its token classes.

Unknown models leave the call unpriced — never a fabricated number. The
"this call was unpriced" signal travels on Cost.unpriced (and the Report's
unpriced_calls/unpriced_models aggregation), not a raw Python warning: the
renderers own how that is shown to a human.
"""

from __future__ import annotations

from datetime import datetime, timezone

from wattage.models import Cost, LLMCall
from wattage.pricing.registry import ModelPrice, PricingRegistry, UnknownModelError


def _call_timestamp(call: LLMCall) -> datetime | None:
    """The call's own wall-clock time, for dated rate windows. end_ns is the
    provider-billed moment; a missing timestamp (0) yields None and the
    registry prices at the currently-effective window."""
    ns = call.end_ns or call.start_ns
    if ns <= 0:
        return None
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


class PricingEngine:
    def __init__(self, registry: PricingRegistry) -> None:
        self.registry = registry

    def price_call(self, call: LLMCall) -> Cost:
        usage = call.usage
        # Tier selection is by prompt-side tokens: the providers that tier
        # (Gemini, Grok) key the rate on the request's prompt length, which
        # includes cached content.
        prompt_tokens = usage.input + usage.cache_read + usage.cache_creation
        try:
            price, priced_as = self.registry.resolve(
                call.provider,
                call.model,
                prompt_tokens=prompt_tokens,
                at=_call_timestamp(call),
            )
        except UnknownModelError:
            return Cost(pricing_version=self.registry.version, unpriced=True)
        if price.unpriced:
            # A published-but-rateless range (e.g. OpenAI's >272K context,
            # whose rate isn't on the pricing page): unpriced, not guessed.
            return Cost(pricing_version=self.registry.version, unpriced=True)

        input_cost = usage.input * price.input
        # usage.output is the visible (non-reasoning) portion only --
        # normalize.py already splits a provider's raw, reasoning-inclusive
        # output_tokens count into these two fields, so adding them here is
        # correct, not a double-charge (both are billed at the output rate).
        output_cost = usage.output * price.output
        cache_read_cost = usage.cache_read * price.cache_read_rate()
        cache_creation_cost = self._cache_write_cost(call, price)
        reasoning_cost = usage.reasoning * price.output
        total = input_cost + output_cost + cache_read_cost + cache_creation_cost + reasoning_cost

        return Cost(
            input=input_cost,
            output=output_cost,
            cache_read=cache_read_cost,
            cache_creation=cache_creation_cost,
            reasoning=reasoning_cost,
            total=total,
            pricing_version=self.registry.version,
            priced_as=priced_as if priced_as != call.model else "",
        )

    @staticmethod
    def _cache_write_cost(call: LLMCall, price: ModelPrice) -> float:
        """Cache-write dollars, using the real per-TTL split when the source
        carries one (Claude Code session logs report 5m/1h separately; the
        gen_ai.* attributes can't). Falls back to the default-TTL rate."""
        usage = call.usage
        breakdown = call.cache_write_ttl_breakdown
        if (
            breakdown
            and price.cache_write_mult_1h is not None
            and price.cache_write_per_token is None
            and breakdown.get("5m", 0) + breakdown.get("1h", 0) == usage.cache_creation
        ):
            return (
                breakdown.get("5m", 0) * price.input * price.cache_write_mult
                + breakdown.get("1h", 0) * price.input * price.cache_write_mult_1h
            )
        return usage.cache_creation * price.cache_write_rate()
