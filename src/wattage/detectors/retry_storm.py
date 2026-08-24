"""retry_storm: the same request re-sent back-to-back — a client retry loop
billing the full prompt every attempt.

The signature is structural: a run of consecutive same-model calls in one
task whose prompt-side token counts all stay within `shape_tolerance`
(default 2%) of the run's FIRST call — a retry resends the same context,
give or take an appended error message — with no tool activity between
attempts (a tool call between two LLM calls means the agent acted on the
response: a healthy iterating loop, not a retry). Anchoring the tolerance
to the first call stops it chaining: a long cached session whose context
grows ~1% per turn drifts arbitrarily far from where it started and must
never read as one endless storm. Three or more matching attempts is not
what a healthy agent step looks like; it's a 429/timeout/schema-error
loop, and every attempt after the first billed the whole prompt again.

The dollars reported are the real billed cost of the repeat attempts (not a
chars/4 projection) — but whether those attempts were *retries* rather than
deliberate same-shaped calls is a heuristic call, so the finding's basis is
`estimated`: visible in every report, never able to fail a CI build on its
own. If the pattern is deliberate in your agent (e.g. a fixed-prompt
sampling loop), disable the detector or raise min_repeats.
"""

from __future__ import annotations

from bisect import bisect_right

from wattage.detectors.base import AnalysisContext, ordered_llm_calls, ordered_tool_calls
from wattage.models import Finding, FindingBasis, LLMCall, QualityRisk, Session, Severity


def _prompt_side(call: LLMCall) -> int:
    return call.usage.input + call.usage.cache_read + call.usage.cache_creation


def _same_shape(a: LLMCall, b: LLMCall, tolerance: float) -> bool:
    if a.provider != b.provider or a.model != b.model:
        return False
    pa, pb = _prompt_side(a), _prompt_side(b)
    if pa == 0 or pb == 0:
        return False
    return abs(pa - pb) / max(pa, pb) <= tolerance


def _tools_between(tool_starts: list[int], prev: LLMCall, curr: LLMCall) -> bool:
    """Any tool activity starting between the two calls?"""
    lo = prev.start_ns
    hi = curr.start_ns
    if hi <= lo:
        return False
    lo_idx = bisect_right(tool_starts, lo)
    return lo_idx < len(tool_starts) and tool_starts[lo_idx] < hi


class RetryStormDetector:
    id = "retry_storm"
    default_enabled = True

    def analyze(self, session: Session, ctx: AnalysisContext) -> list[Finding]:
        cfg = ctx.config.detectors.retry_storm
        if not cfg.enabled:
            return []

        findings: list[Finding] = []
        for task in session.tasks:
            calls = ordered_llm_calls(task)
            # A retry loop cannot execute tools between attempts — a tool
            # call between two LLM calls means the agent acted on the first
            # response, which is exactly what a healthy iterating loop looks
            # like. Any intervening tool activity breaks the run.
            tool_starts = sorted(tc.start_ns for tc in ordered_tool_calls(task))
            run: list[LLMCall] = []
            for call in calls:
                if (
                    run
                    # Anchor comparison: every attempt must match the run's
                    # FIRST call, not merely its predecessor — comparing to
                    # run[-1] lets the 2% tolerance chain, so a long cached
                    # session whose context grows ~1%/turn would read as one
                    # endless "storm" despite drifting arbitrarily far from
                    # where it started.
                    and _same_shape(run[0], call, cfg.shape_tolerance)
                    and not _tools_between(tool_starts, run[-1], call)
                ):
                    run.append(call)
                    continue
                if len(run) >= cfg.min_repeats:
                    findings.append(self._finding_for_run(run, task.task_id))
                run = [call]
            if len(run) >= cfg.min_repeats:
                findings.append(self._finding_for_run(run, task.task_id))
        return findings

    def _finding_for_run(self, run: list[LLMCall], task_id: str) -> Finding:
        repeats = run[1:]
        wasted_tokens = sum(c.usage.total() for c in repeats)
        wasted_dollars = sum(c.cost.total for c in repeats)
        severity = Severity.high if len(run) >= 5 else Severity.medium
        return Finding(
            id=self.id,
            severity=severity,
            wasted_tokens=wasted_tokens,
            wasted_dollars=wasted_dollars,
            basis=FindingBasis.estimated,
            quality_risk=QualityRisk.none,
            evidence=(
                f"{len(run)} consecutive {run[0].model} calls in task {task_id} with "
                f"near-identical prompt sizes (~{_prompt_side(run[0])} prompt-side tokens "
                "each) — the shape of a retry loop re-billing the same context every attempt"
            ),
            fix=(
                "Add backoff with a retry cap, and fix the underlying failure (schema "
                "validation, rate limit, timeout) instead of resending the full prompt. "
                "If this pattern is deliberate, raise retry_storm.min_repeats or disable "
                "the detector."
            ),
            span_ids=[c.span_id for c in run],
        )
