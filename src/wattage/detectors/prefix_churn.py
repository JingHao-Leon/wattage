"""prefix_churn (doc §4.1): re-sent context not being cached.

Per task, walks LLM calls in chronological order. Without message content
capture we can't diff prompts directly, so this uses the token-count proxy
the doc allows as a fallback: if a call's input tokens are at least as large
as the *entire* prior call's context (input+output) and no cache read
happened on this call, that prior context was almost certainly re-sent
verbatim and billed at full input price. The monotonic-growth guard
(curr.input >= prior_total) is what keeps this from flagging a genuinely
different/shrunk prefix.

Known limitation: two coincidentally same-sized-but-different prefixes can't
be told apart from token counts alone — that needs message capture or a
prompt fingerprint (see LLMCall.prompt_fingerprint), which isn't populated
yet. Left as a documented future enhancement rather than faked here.

Finding.wasted_dollars is the *recoverable* amount, not the full resent
cost: caching a hit still bills at the model's cached-input rate (a
multiplier or an absolute per-token rate, depending on the registry entry),
so the achievable savings is resent_tokens * (input rate - cached rate),
matching benchmarks/frontier.py's real-data before/after simulation. Severity stays
keyed on the gross resent_dollars share of task cost (doc §4.1) -- how much
of the task's spend the churn touches at all, not how much of that is
ultimately recoverable.

find_resent_segments also filters out a resend below the model's own
min_cacheable_prefix_tokens (vendored per-model in the pricing registry):
below that size, the provider won't create a cache entry for it at all, so
"enable prompt caching" -- the fix this detector recommends -- genuinely
would not help. Flagging it anyway would be confident, specific, actionable
advice that doesn't work, which is worse than not flagging it.
"""

from __future__ import annotations

from wattage.detectors.base import AnalysisContext, ordered_llm_calls, rate_card_for
from wattage.models import Finding, LLMCall, QualityRisk, Session, Severity, Task
from wattage.pricing.registry import PricingRegistry, UnknownModelError


def find_resent_segments(task: Task, registry: PricingRegistry) -> list[tuple[LLMCall, int]]:
    """(call, resent_tokens) pairs where `call` almost certainly re-sent the
    entire prior call's context uncached — the pure detection logic, reused
    by this detector and by benchmarks/frontier.py's real-data caching-fix
    simulation so both share one implementation."""
    calls = ordered_llm_calls(task)
    segments: list[tuple[LLMCall, int]] = []
    for prev, curr in zip(calls, calls[1:], strict=False):
        if curr.usage.cache_read > 0:
            continue  # caching is active for this turn; not churn
        prior_total = prev.usage.input + prev.usage.output
        if prior_total <= 0 or curr.usage.input < prior_total:
            continue  # prefix shrank or changed; not a simple resend
        try:
            # curr is the request that would need to mark this prefix
            # cacheable, so curr's own model/provider is what governs
            # whether that provider's minimum cacheable size is met.
            price = rate_card_for(registry, curr)
        except UnknownModelError:
            segments.append((curr, prior_total))  # can't check the threshold; don't guess
            continue
        if prior_total < price.min_cacheable_prefix_tokens:
            continue  # too small for this provider to cache at all -- not a fixable gap
        segments.append((curr, prior_total))
    return segments


class PrefixChurnDetector:
    id = "prefix_churn"
    default_enabled = True

    def analyze(self, session: Session, ctx: AnalysisContext) -> list[Finding]:
        cfg = ctx.config.detectors.prefix_churn
        findings: list[Finding] = []

        for task in session.tasks:
            segments = find_resent_segments(task, ctx.pricing.registry)
            if not segments:
                continue

            resent_tokens = 0
            resent_dollars = 0.0
            recoverable_dollars = 0.0
            span_ids: list[str] = []
            for call, tokens in segments:
                try:
                    price = rate_card_for(ctx.pricing.registry, call)
                except UnknownModelError:
                    continue
                resent_tokens += tokens
                resent_dollars += tokens * price.input
                # The fix (prompt caching) doesn't make a re-sent prefix free --
                # a cache hit still bills at the model's cached-input rate
                # (0.1x input for Anthropic/Mistral, but an absolute per-token
                # rate on most OpenAI/Google/xAI entries, ranging 0.1x-0.5x of
                # input). price.cache_read_rate() resolves whichever form the
                # registry entry uses; computing this from cache_read_mult
                # alone would silently zero the savings for absolute-rate
                # models (mult defaults to 1.0 there). Crediting the full
                # resent_dollars as "wasted" would overstate the achievable
                # savings; benchmarks/frontier.py's real-data before/after
                # simulation accounts for it the same way.
                recoverable_dollars += tokens * max(price.input - price.cache_read_rate(), 0.0)
                span_ids.append(call.span_id)

            if resent_tokens == 0:
                continue

            calls = ordered_llm_calls(task)
            task_dollars = sum(c.cost.total for c in calls)
            # Severity is about how much of this task's spend went to resending
            # stale context at all -- the size of the problem -- so it stays
            # keyed on the gross resent_dollars, independent of what fraction
            # of that is actually recoverable via caching.
            ratio = resent_dollars / task_dollars if task_dollars > 0 else 0.0
            severity = Severity.high if ratio >= cfg.high_severity_ratio else Severity.medium

            findings.append(
                Finding(
                    id=self.id,
                    severity=severity,
                    wasted_tokens=resent_tokens,
                    wasted_dollars=recoverable_dollars,
                    quality_risk=QualityRisk.none,
                    evidence=(
                        f"{resent_tokens} tokens of prior context re-sent uncached across "
                        f"{len(span_ids)} turn(s) in task {task.task_id} "
                        f"({ratio:.0%} of task cost)"
                    ),
                    fix=(
                        "Enable prompt caching on the stable prefix (system prompt + tool "
                        "schemas); move volatile content to the tail of the prompt."
                    ),
                    span_ids=span_ids,
                )
            )

        return findings
