"""verbosity (doc §4.6): output over-generation.

Flags calls with no max_tokens cap whose output exceeds a configured
expected-output ceiling. This is deliberately a single global ceiling, not
the doc's per-task-type bands (extract: 0.3x, classify: 0.1x, draft: 3.0x
input) — those need task-type classification, which doesn't exist in the
model yet (LLMCall has no task-type field), and guessing at task type would
risk flagging legitimate long-form generation. A call with an explicit
max_tokens is never flagged: the developer already made a deliberate choice
about the ceiling, and this detector can't second-guess it without knowing
whether the output was substantively useful.

"wasted_tokens" here means "tokens beyond the configured ceiling," a policy
estimate, not a factual claim that those specific tokens were unnecessary —
that would require judging the actual content, an out-of-scope-for-Phase-1
capability (see doc §12.4's LLM-judge, gated off by default).
"""

from __future__ import annotations

from wattage.detectors.base import AnalysisContext, ordered_llm_calls, rate_card_for
from wattage.models import Finding, FindingBasis, QualityRisk, Session, Severity
from wattage.pricing.registry import UnknownModelError


class VerbosityDetector:
    id = "verbosity"
    default_enabled = True

    def analyze(self, session: Session, ctx: AnalysisContext) -> list[Finding]:
        cfg = ctx.config.detectors.verbosity
        findings: list[Finding] = []

        for task in session.tasks:
            for call in ordered_llm_calls(task):
                if call.max_tokens is not None:
                    continue  # a cap is set; not this detector's concern
                if call.usage.output <= cfg.expected_output_ceiling:
                    continue

                try:
                    price = rate_card_for(ctx.pricing.registry, call)
                except UnknownModelError:
                    continue

                excess = call.usage.output - cfg.expected_output_ceiling
                wasted_dollars = excess * price.output
                high_threshold = cfg.expected_output_ceiling * cfg.high_severity_multiplier
                severity = Severity.high if call.usage.output >= high_threshold else Severity.medium

                findings.append(
                    Finding(
                        id=self.id,
                        severity=severity,
                        wasted_tokens=excess,
                        wasted_dollars=wasted_dollars,
                        basis=FindingBasis.estimated,
                        quality_risk=QualityRisk.low,
                        evidence=(
                            f"{call.usage.output} output tokens with no max_tokens cap set "
                            f"(configured expected ceiling: {cfg.expected_output_ceiling})"
                        ),
                        fix=(
                            "Prompt for concise or structured/extractive output, or use the "
                            "provider's effort/verbosity control for this step. A max_tokens "
                            "cap also works, but set it generously — a cap tight enough to "
                            "truncate mid-answer forces retries that cost more than the "
                            "excess."
                        ),
                        span_ids=[call.span_id],
                    )
                )

        return findings
