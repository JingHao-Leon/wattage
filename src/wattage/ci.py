"""wattage ci (doc §11): the cost-regression gate.

Exit codes exactly per §11.3:
  0 pass
  1 fail — a fail-on threshold breached
  2 config/usage error
  3 ingestion error — unparseable *or empty* trace (the doc names both)
  4 pricing error — at least one call the pricing engine couldn't price;
    an incomplete cost figure can't honestly gate anything, so this fails
    loudly rather than silently comparing against an undercount.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml
from pydantic import ValidationError

from wattage.adapters.base import AdapterError
from wattage.baseline import (
    Baseline,
    cost_delta_pct,
    load_baseline,
    record_run,
    save_baseline,
)
from wattage.config import CIFailOnConfig, WattageConfig
from wattage.models import Report, Severity
from wattage.pricing.registry import PricingConfigError, PricingRegistry
from wattage.report import build_trace_and_report, load_quality_map
from wattage.scoring.quality import QualityMapError

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CONFIG_ERROR = 2
EXIT_INGESTION_ERROR = 3
EXIT_PRICING_ERROR = 4


class CIConfigError(Exception):
    """Exit code 2."""


@dataclass
class CIResult:
    exit_code: int
    report: Report | None = None
    baseline: Baseline | None = None
    reasons: list[str] = field(default_factory=list)
    # Non-failure context for the verdict UI: baseline initialization,
    # what the run was compared against, pricing-version annotations.
    notes: list[str] = field(default_factory=list)
    baseline_created: bool = False


_FAIL_ON_KEYS = frozenset({"score_below", "cost_delta_pct_above", "any_critical"})


def parse_fail_on(spec: str) -> CIFailOnConfig:
    """Parses "key:value,key:value" — the doc's own action.yml example format,
    e.g. "score_below:80,cost_delta_pct_above:5,any_critical:true"."""
    values: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise CIConfigError(f"invalid --fail-on clause (expected key:value): {part!r}")
        key, _, value = part.partition(":")
        key = key.strip()
        if key not in _FAIL_ON_KEYS:
            raise CIConfigError(
                f"unrecognized --fail-on key {key!r} (expected one of "
                f"{', '.join(sorted(_FAIL_ON_KEYS))})"
            )
        values[key] = value.strip()

    defaults = CIFailOnConfig()
    try:
        score_below = (
            int(values["score_below"]) if "score_below" in values else defaults.score_below
        )
        cost_delta_pct_above = (
            float(values["cost_delta_pct_above"])
            if "cost_delta_pct_above" in values
            else defaults.cost_delta_pct_above
        )
    except ValueError as exc:
        raise CIConfigError(f"invalid --fail-on numeric value: {exc}") from exc
    any_critical = (
        values["any_critical"].strip().lower() in ("true", "1", "yes")
        if "any_critical" in values
        else defaults.any_critical
    )
    return CIFailOnConfig(
        score_below=score_below,
        cost_delta_pct_above=cost_delta_pct_above,
        any_critical=any_critical,
    )


def evaluate_fail_on(
    report: Report,
    baseline: Baseline,
    fail_on: CIFailOnConfig,
    notes: list[str] | None = None,
) -> list[str]:
    reasons = []
    if fail_on.score_below is not None and report.score.efficiency < fail_on.score_below:
        reasons.append(f"score {report.score.efficiency} is below threshold {fail_on.score_below}")
    if fail_on.cost_delta_pct_above is not None:
        last = baseline.last_passing
        if (
            last is not None
            and last.pricing_version
            and last.pricing_version != report.pricing_version
        ):
            # The baseline was priced under a different snapshot — a rate
            # change is indistinguishable from a real regression in raw
            # dollars, so the delta gate is skipped (announced, not silent)
            # and the baseline refreshes under the new snapshot on this run.
            if notes is not None:
                notes.append(
                    f"pricing snapshot changed ({last.pricing_version} → "
                    f"{report.pricing_version}); cost-delta gate skipped — raw dollars "
                    "across snapshots mix a rate change with a real usage change. The "
                    "gate resumes once a PASSING run records a baseline under the new "
                    "snapshot; until then, review cost changes by hand."
                )
        else:
            delta = cost_delta_pct(report, baseline)
            if delta is not None and delta > fail_on.cost_delta_pct_above:
                reasons.append(
                    f"cost increased {delta:.1f}% vs baseline, above threshold "
                    f"{fail_on.cost_delta_pct_above}%"
                )
    if fail_on.any_critical and any(f.severity == Severity.critical for f in report.findings):
        reasons.append("at least one critical-severity finding")
    return reasons


def run_ci(
    source: str,
    baseline_path: str | None = None,
    pricing_override: str | None = None,
    quality_file: str | None = None,
    fail_on: CIFailOnConfig | None = None,
    config: WattageConfig | None = None,
    update_baseline: bool = True,
) -> CIResult:
    config = config or WattageConfig()
    fail_on = fail_on or config.ci.fail_on
    resolved_baseline_path = baseline_path or config.ci.baseline_path

    # Config inputs are loaded BEFORE the trace: a missing or malformed
    # --quality/--pricing file is a config/usage problem (exit 2), and
    # loading them inside trace ingestion used to misreport it as exit 3
    # ("unparseable trace") — sending users to debug the wrong artifact.
    try:
        quality_map = load_quality_map(quality_file)
    except (OSError, json.JSONDecodeError, ValueError, QualityMapError) as exc:
        return CIResult(
            exit_code=EXIT_CONFIG_ERROR,
            reasons=[f"could not read --quality file {quality_file!r}: {exc}"],
        )
    try:
        registry = PricingRegistry.load(overrides_path=pricing_override)
    except (OSError, yaml.YAMLError, PricingConfigError, TypeError) as exc:
        detail = f" override {pricing_override!r}" if pricing_override else ""
        return CIResult(
            exit_code=EXIT_CONFIG_ERROR,
            reasons=[f"could not load pricing{detail}: {exc}"],
        )

    try:
        trace, report = build_trace_and_report(
            source,
            config=config,
            registry=registry,
            quality_map=quality_map,
        )
    except (OSError, json.JSONDecodeError, AdapterError) as exc:
        # AdapterError also covers a structurally-valid but empty trace
        # (zero sessions) -- build_trace_and_report raises it for that
        # case itself now, so there's exactly one place that decides what
        # counts as "nothing to grade" instead of two copies that could
        # drift apart.
        return CIResult(exit_code=EXIT_INGESTION_ERROR, reasons=[str(exc)])
    except (yaml.YAMLError, PricingConfigError, QualityMapError) as exc:
        # A malformed quality-map shape or bad vendored pricing entry is a
        # config/usage problem, not a trace-ingestion one -- same exit code
        # as a bad --config file, not exit 3.
        return CIResult(exit_code=EXIT_CONFIG_ERROR, reasons=[str(exc)])

    if report.unpriced_calls > 0:
        models = ", ".join(report.unpriced_models)
        return CIResult(
            exit_code=EXIT_PRICING_ERROR,
            report=report,
            reasons=[
                f"{report.unpriced_calls} call(s) had no pricing entry ({models}) — "
                "supply a --pricing override or the cost comparison would be an undercount"
            ],
        )

    notes: list[str] = []
    try:
        # The baseline as it stood *before* this run — what this run was
        # actually compared against, and what render_pr_comment/render's
        # delta table should diff against. Comparing against the
        # post-update baseline would (once this run itself becomes
        # last_passing) show the run diffed against itself: always "no
        # change".
        baseline_before = load_baseline(resolved_baseline_path)
        baseline_created = baseline_before.last_passing is None
        reasons = evaluate_fail_on(report, baseline_before, fail_on, notes=notes)
        passed = not reasons
        exit_code = EXIT_PASS if passed else EXIT_FAIL

        if update_baseline:
            updated = record_run(
                baseline_before, report, passed=passed, window_days=config.ci.rolling_window_days
            )
            save_baseline(resolved_baseline_path, updated)
    except (OSError, ValidationError, ValueError, TypeError) as exc:
        # baseline.json is a plain committed file a user (or a bad merge)
        # can hand-corrupt -- a config-level problem, not a real cost
        # regression. Letting this fall through uncaught would land on
        # Python's default exit code 1, indistinguishable from exit 1's
        # real documented meaning ("a fail-on threshold breached"): a
        # corrupt baseline would masquerade as a genuine regression.
        # TypeError included: a hand-edited naive timestamp used to escape
        # as an uncaught aware-vs-naive comparison crash.
        return CIResult(
            exit_code=EXIT_CONFIG_ERROR,
            reasons=[f"could not read/update baseline {resolved_baseline_path!r}: {exc}"],
        )

    return CIResult(
        exit_code=exit_code,
        report=report,
        baseline=baseline_before,
        reasons=reasons,
        notes=notes,
        baseline_created=baseline_created,
    )
