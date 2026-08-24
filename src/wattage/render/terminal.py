"""Terminal renderer (doc §7.7): compact rich summary.

Quality is honestly reported as unmeasured until a --quality map is supplied
rather than guessed at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.table import Table

from wattage.models import FindingBasis, Report
from wattage.render.format import format_dollars

if TYPE_CHECKING:
    from wattage.ci import CIResult

# A bare Console() auto-detects the live terminal's width, so the exact same
# report renders at a different box width for every user/environment (and
# differently again for a non-tty/piped invocation) -- undermining the
# "share a screenshot, docs show exactly this" premise the report/README are
# built around. Pinned to a fixed width instead, so output is reproducible
# regardless of the caller's terminal.
_DEFAULT_WIDTH = 100

_GRADE_STYLE = {"A": "bold green", "B": "bold green", "C": "bold yellow", "D": "bold red"}
_SEVERITY_STYLE = {
    "critical": "bold red",
    "high": "red",
    "medium": "yellow",
    "low": "dim",
    "info": "dim",
}


def _grade_style(grade: str) -> str:
    return _GRADE_STYLE.get(grade, "bold red")


def render_terminal(report: Report, console: Console | None = None) -> None:
    console = console or Console(width=_DEFAULT_WIDTH)

    grade_markup = (
        f"[{_grade_style(report.score.grade)}]{report.score.grade} ({report.score.efficiency})[/]"
    )
    headline = (
        f"[bold]Token Efficiency:[/bold] {grade_markup}"
        f"   [bold]Total cost:[/bold] {format_dollars(report.total_dollars)}"
    )
    if report.score.monthly_projection is not None:
        headline += (
            f"\n[bold]~{format_dollars(report.score.monthly_projection)}/mo recoverable[/bold] "
            "at your stated run rate"
        )
    if not report.score.quality_measured:
        headline += "\n[dim]quality: unmeasured[/dim]"
    if report.source_format == "claude-code":
        headline += (
            "\n[dim]Claude Code session — costs shown at standard Anthropic API rates "
            "(the API-equivalent value on a subscription plan)[/dim]"
        )
    if report.unpriced_calls:
        # Model names come straight from trace content — escaped so a
        # bracketed name can neither crash rich nor restyle the report.
        models = rich_escape(", ".join(report.unpriced_models))
        warning = (
            f"[bold yellow]⚠ {report.unpriced_calls} call(s) unpriced[/bold yellow] "
            f"(no registry entry for {models}) — cost and score below are incomplete"
        )
        headline = f"{warning}\n{headline}"
    title = f"⚡ wattage — {rich_escape(report.trace_source)}"
    if len(title) > _DEFAULT_WIDTH - 10:
        # Long paths truncate mid-filename inside a panel title; show the
        # basename there and keep the full path legible in the body.
        basename = report.trace_source.rsplit("/", 1)[-1]
        title = f"⚡ wattage — {rich_escape(basename)}"
        headline = f"[dim]{rich_escape(report.trace_source)}[/dim]\n{headline}"
    console.print(Panel(headline, title=title, expand=False))

    if report.dropped_token_spans or report.duplicate_spans or report.skipped_spans:
        parts = []
        if report.dropped_token_spans:
            parts.append(
                f"{report.dropped_token_spans} span(s) carried token usage but no "
                "recognizable call kind and were NOT counted"
            )
        if report.duplicate_spans:
            parts.append(
                f"{report.duplicate_spans} duplicate span id(s) collapsed to their first occurrence"
            )
        if report.skipped_spans:
            parts.append(f"{len(report.skipped_spans)} span(s) skipped as malformed")
        console.print(
            f"[bold yellow]⚠ coverage:[/bold yellow] {'; '.join(parts)} — totals may "
            "be an undercount"
        )

    table = Table(title="Token breakdown")
    table.add_column("Category")
    table.add_column("Tokens", justify="right")
    for category, tokens in report.token_breakdown.items():
        table.add_row(category, f"{tokens:,}")
    console.print(table)

    if report.findings:
        findings_table = Table(title="Findings")
        findings_table.add_column("Detector")
        findings_table.add_column("Severity")
        findings_table.add_column("Wasted $", justify="right")
        findings_table.add_column("Fix")
        for f in report.findings:
            severity_markup = f"[{_SEVERITY_STYLE.get(f.severity.value, '')}]{f.severity.value}[/]"
            wasted = format_dollars(f.wasted_dollars)
            if f.basis == FindingBasis.estimated:
                wasted = f"~{wasted}"
            findings_table.add_row(rich_escape(f.id), severity_markup, wasted, rich_escape(f.fix))
        console.print(findings_table)
        measured_total = report.score.recoverable_dollars
        estimated_total = report.score.estimated_extra_dollars
        summary = f"measured waste: {format_dollars(measured_total)} (counts toward the grade)"
        if estimated_total > 0:
            summary += (
                f" · estimated (~) findings: {format_dollars(estimated_total)} "
                "(reported, never graded)"
            )
        console.print(f"[dim]{summary}[/dim]")
    elif report.unpriced_calls:
        console.print(
            "[dim]No findings on the priced portion of this trace — "
            f"{report.unpriced_calls} call(s) could not be priced, so this is not "
            "a confirmed-efficient trace.[/dim]"
        )
    else:
        console.print("[dim]No findings — this trace looks efficient.[/dim]")

    console.print(f"[dim]pricing: {report.pricing_version}[/dim]")


def render_ci_verdict(
    result: CIResult,
    baseline_path: str,
    baseline_updated: bool,
    console: Console | None = None,
) -> None:
    """A gate needs a verdict you can't miss: an explicit PASS/FAIL banner,
    where the baseline came from, and what (if anything) was written."""
    console = console or Console(width=_DEFAULT_WIDTH, stderr=result.exit_code != 0)

    for note in result.notes:
        console.print(f"[yellow]note:[/yellow] {rich_escape(note)}")

    if result.exit_code == 0:
        if result.baseline_created and baseline_updated:
            console.print(
                f"[yellow]note:[/yellow] no baseline found — initialized {baseline_path} "
                "(this run becomes the baseline for future comparisons)"
            )
        elif result.baseline is not None and result.baseline.last_passing is not None:
            last = result.baseline.last_passing
            console.print(
                f"[dim]compared against baseline from {last.recorded_at} "
                f"(score {last.efficiency}, {format_dollars(last.total_dollars)})[/dim]"
            )
        if not baseline_updated:
            console.print("[dim]dry run: baseline file left untouched (--no-update-baseline)[/dim]")
        console.print("[bold green]✓ PASS[/bold green] — no cost regression")
        return

    for reason in result.reasons:
        # Reasons embed trace-derived text (adapter errors quote bad field
        # values; the exit-4 reason names unpriced models) — escaped, or a
        # bracketed value would raise MarkupError and turn a documented
        # exit 3/4 into a traceback with Python's default exit 1, the exact
        # "fake cost regression" the exit-code contract exists to prevent.
        console.print(f"[red]- {rich_escape(reason)}[/red]")
    if result.exit_code == 1:
        console.print("[bold red]✗ FAIL[/bold red] — cost-regression gate tripped")
    else:
        label = {2: "config error", 3: "ingestion error", 4: "pricing error"}.get(
            result.exit_code, "error"
        )
        console.print(f"[bold red]✗ {label}[/bold red] (exit {result.exit_code})")
