import json
import warnings as _warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated, TypeVar

import typer
import yaml

from wattage import __version__
from wattage.adapters.base import AdapterError
from wattage.adapters.claude_code import find_latest_session
from wattage.ci import CIConfigError, parse_fail_on, run_ci
from wattage.config import WattageConfig, WattageConfigError, load_config
from wattage.pricing.registry import PricingConfigError
from wattage.render.badge import render_badge
from wattage.render.format import format_dollars
from wattage.render.html import render_html
from wattage.render.json_report import render_json
from wattage.render.junit import render_junit
from wattage.render.pr_comment import render_pr_comment
from wattage.render.sarif import render_sarif
from wattage.render.terminal import render_ci_verdict, render_terminal
from wattage.report import build_report, build_trace_and_report
from wattage.scoring.quality import QualityMapError

app = typer.Typer(
    name="wattage",
    help="A token-spend profiler and cost-regression gate for AI agents.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"wattage {__version__}")
        raise typer.Exit()


_CONFIG_HELP = "Path to a wattage.yaml config file. Defaults to ./wattage.yaml if present."
_SOURCE_HELP = (
    "Path to a trace: OTLP JSON (single-object or JSON Lines) or a Claude Code "
    "session .jsonl / project directory — the format is auto-detected."
)
_CLAUDE_CODE_HELP = (
    "Analyze your latest Claude Code session (from ~/.claude/projects) when no "
    "source path is given."
)
_RUNS_HELP = (
    "Project recoverable dollars to a monthly figure at this many runs/month. "
    "Only your own stated run rate is ever used — never an extrapolation from "
    "the trace's duration."
)


def _load_config_or_exit(config_path: Path | None) -> WattageConfig:
    try:
        return load_config(str(config_path) if config_path else None)
    except WattageConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


def _resolve_source(source: Path | None, claude_code: bool) -> Path:
    """A source path wins; --claude-code with no path finds the newest
    session; neither is a usage error with a pointer to both options."""
    if source is not None:
        return source
    if claude_code:
        latest = find_latest_session()
        if latest is None:
            typer.echo(
                "error: no Claude Code sessions found under ~/.claude/projects — "
                "has Claude Code run on this machine?",
                err=True,
            )
            raise typer.Exit(code=2)
        typer.echo(f"analyzing latest Claude Code session: {latest}", err=True)
        return latest
    typer.echo(
        "error: give a trace path, or pass --claude-code to analyze your latest "
        "Claude Code session. Try `wattage demo` to see a sample report first.",
        err=True,
    )
    raise typer.Exit(code=2)


_T = TypeVar("_T")


@contextmanager
def _styled_warnings() -> Iterator[None]:
    """Library-level UserWarnings (embedder fallback, etc.) render as clean
    one-line notes after the report, instead of leaking raw Python warning
    text with site-packages paths mid-output."""
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        try:
            yield
        finally:
            # Flush even when the wrapped code raises (typer.Exit on an
            # input error) — record=True suppresses the default stderr
            # passthrough, so skipping this would silently swallow the
            # warnings on exactly the error paths.
            seen: set[str] = set()
            for warning in caught:
                message = str(warning.message)
                if message not in seen:
                    seen.add(message)
                    typer.secho(f"note: {message}", err=True, fg=typer.colors.YELLOW)


def _input_errors_or_exit(build: Callable[[], _T], source: Path | None = None) -> _T:
    """report/score/badge have no documented multi-code exit contract like
    `ci` does, so a bad --source/--pricing/--quality file just needs a
    clean message instead of `ci`'s carefully-scoped 0-4 codes -- a raw
    Python traceback for a missing or malformed file is not launch-quality
    UX, and `ci` already gets this right for the same underlying errors.
    """
    prefix = f"{source}: " if source is not None else ""
    try:
        return build()
    except FileNotFoundError as exc:
        typer.echo(f"error: {prefix}file not found ({exc.filename})", err=True)
        raise typer.Exit(code=1) from exc
    except json.JSONDecodeError as exc:
        typer.echo(
            f"error: {prefix}not valid JSON (parse error at line {exc.lineno}, "
            f"column {exc.colno}). Wattage reads OTLP JSON trace exports and Claude "
            "Code session files — see https://faizannraza.github.io/wattage/getting-started/",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except (
        OSError,
        yaml.YAMLError,
        AdapterError,
        PricingConfigError,
        QualityMapError,
    ) as exc:
        typer.echo(f"error: {prefix}{exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """wattage: see where your tokens burn."""


@app.command()
def report(
    source: Annotated[Path | None, typer.Argument(help=_SOURCE_HELP)] = None,
    claude_code: Annotated[bool, typer.Option("--claude-code", help=_CLAUDE_CODE_HELP)] = False,
    pricing: Annotated[Path | None, typer.Option(help="Path to a pricing.yaml override.")] = None,
    quality: Annotated[
        Path | None,
        typer.Option(help="Path to a quality.json map; enables quality-gated findings."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the full machine-readable JSON report.")
    ] = False,
    html_output: Annotated[
        Path | None,
        typer.Option("--html", help="Write a self-contained HTML flame graph."),
    ] = None,
    runs_per_month: Annotated[int | None, typer.Option("--runs-per-month", help=_RUNS_HELP)] = None,
    config: Annotated[Path | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
) -> None:
    """Ingest a trace and print a priced, findings-quantified report."""
    cfg = _load_config_or_exit(config)
    resolved_source = _resolve_source(source, claude_code)
    with _styled_warnings():
        trace, report_obj = _input_errors_or_exit(
            lambda: build_trace_and_report(
                str(resolved_source),
                pricing_override=str(pricing) if pricing else None,
                config=cfg,
                quality_file=str(quality) if quality else None,
                runs_per_month=runs_per_month,
            ),
            source=resolved_source,
        )
        if html_output is not None:
            html_output.write_text(render_html(trace, report_obj), encoding="utf-8")
            # Status line goes to stderr when stdout is the JSON stream.
            typer.echo(f"wrote {html_output}", err=json_output)
            if not json_output:
                return
        if json_output:
            typer.echo(render_json(report_obj))
        else:
            render_terminal(report_obj)


@app.command()
def demo() -> None:
    """Run the report on a bundled findings-rich sample trace — no setup,
    no trace of your own needed."""
    demo_trace = resources.files("wattage.data").joinpath("demo_trace.json")
    with resources.as_file(demo_trace) as path:
        with _styled_warnings():
            report_obj = _input_errors_or_exit(lambda: build_report(str(path)))
            # The bundled trace's install path is noise; the report should
            # identify the demo, not the wheel's cache directory.
            report_obj.trace_source = "demo_trace.json (bundled)"
            render_terminal(report_obj)
    typer.echo("")
    typer.secho("That was a bundled demo trace. Try your own data next:", fg=typer.colors.CYAN)
    typer.echo("  wattage report --claude-code        # your latest Claude Code session")
    typer.echo("  wattage report path/to/trace.json   # any OTLP GenAI trace export")


@app.command()
def score(
    source: Annotated[Path | None, typer.Argument(help=_SOURCE_HELP)] = None,
    claude_code: Annotated[bool, typer.Option("--claude-code", help=_CLAUDE_CODE_HELP)] = False,
    pricing: Annotated[Path | None, typer.Option(help="Path to a pricing.yaml override.")] = None,
    quality: Annotated[
        Path | None,
        typer.Option(help="Path to a quality.json map; enables quality-gated findings."),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the score as machine-readable JSON.")
    ] = False,
    runs_per_month: Annotated[int | None, typer.Option("--runs-per-month", help=_RUNS_HELP)] = None,
    config: Annotated[Path | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
) -> None:
    """Print just the Token Efficiency score and dollar headline."""
    cfg = _load_config_or_exit(config)
    resolved_source = _resolve_source(source, claude_code)
    with _styled_warnings():
        report_obj = _input_errors_or_exit(
            lambda: build_report(
                str(resolved_source),
                pricing_override=str(pricing) if pricing else None,
                config=cfg,
                quality_file=str(quality) if quality else None,
                runs_per_month=runs_per_month,
            ),
            source=resolved_source,
        )
    if json_output:
        # Same honesty contract as text mode: a grade computed from an
        # incomplete cost figure is never presented as a normal score. The
        # machine-readable shape carries the unpriced signal explicitly,
        # with score null when the grade would be fabricated.
        payload: dict[str, object] = {
            "unpriced_calls": report_obj.unpriced_calls,
            "unpriced_models": report_obj.unpriced_models,
            "score": None if report_obj.unpriced_calls else report_obj.score.model_dump(),
        }
        typer.echo(json.dumps(payload, indent=2))
        return
    if report_obj.unpriced_calls:
        call_word = "call" if report_obj.unpriced_calls == 1 else "calls"
        models = ", ".join(report_obj.unpriced_models)
        typer.echo(
            f"unpriced ({report_obj.unpriced_calls} {call_word}: {models}) — "
            "see `wattage report` for detail"
        )
        return
    s = report_obj.score
    typer.echo(f"{s.grade} ({s.efficiency}) · {format_dollars(s.recoverable_dollars)} recoverable")


@app.command()
def badge(
    source: Annotated[Path | None, typer.Argument(help=_SOURCE_HELP)] = None,
    claude_code: Annotated[bool, typer.Option("--claude-code", help=_CLAUDE_CODE_HELP)] = False,
    pricing: Annotated[Path | None, typer.Option(help="Path to a pricing.yaml override.")] = None,
    quality: Annotated[
        Path | None,
        typer.Option(help="Path to a quality.json map; enables quality-gated findings."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the SVG to this file instead of stdout.")
    ] = None,
    runs_per_month: Annotated[int | None, typer.Option("--runs-per-month", help=_RUNS_HELP)] = None,
    config: Annotated[Path | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
) -> None:
    """Emit a Token Efficiency SVG badge."""
    cfg = _load_config_or_exit(config)
    resolved_source = _resolve_source(source, claude_code)
    with _styled_warnings():
        report_obj = _input_errors_or_exit(
            lambda: build_report(
                str(resolved_source),
                pricing_override=str(pricing) if pricing else None,
                config=cfg,
                quality_file=str(quality) if quality else None,
                runs_per_month=runs_per_month,
            ),
            source=resolved_source,
        )
    svg = render_badge(report_obj)
    if out is not None:
        out.write_text(svg, encoding="utf-8")
    else:
        typer.echo(svg)


@app.command()
def ci(
    source: Annotated[Path | None, typer.Argument(help=_SOURCE_HELP)] = None,
    claude_code: Annotated[bool, typer.Option("--claude-code", help=_CLAUDE_CODE_HELP)] = False,
    baseline: Annotated[
        Path | None, typer.Option(help="Path to .wattage/baseline.json (default from config).")
    ] = None,
    pricing: Annotated[Path | None, typer.Option(help="Path to a pricing.yaml override.")] = None,
    quality: Annotated[
        Path | None,
        typer.Option(help="Path to a quality.json map; enables quality-gated findings."),
    ] = None,
    fail_on: Annotated[
        str | None,
        typer.Option(
            "--fail-on",
            help='e.g. "score_below:80,cost_delta_pct_above:5,any_critical:true".',
        ),
    ] = None,
    update_baseline: Annotated[
        bool,
        typer.Option(
            "--update-baseline/--no-update-baseline",
            help=(
                "Record this run into the baseline file (default). Use "
                "--no-update-baseline for a local dry run that must not touch "
                "the committed baseline."
            ),
        ),
    ] = True,
    pr_comment_out: Annotated[
        Path | None, typer.Option("--pr-comment-out", help="Write the markdown PR comment here.")
    ] = None,
    sarif_out: Annotated[
        Path | None, typer.Option("--sarif-out", help="Write SARIF results here.")
    ] = None,
    junit_out: Annotated[
        Path | None, typer.Option("--junit-out", help="Write JUnit XML results here.")
    ] = None,
    badge_out: Annotated[
        Path | None, typer.Option("--badge-out", help="Write a Token Efficiency SVG badge here.")
    ] = None,
    report_json_out: Annotated[
        Path | None,
        typer.Option(
            "--report-json-out",
            help="Write the full machine-readable JSON report here (for downstream steps).",
        ),
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", help=_CONFIG_HELP)] = None,
) -> None:
    """Cost-regression gate for CI: fails the build on a real regression."""
    cfg = _load_config_or_exit(config)
    resolved_source = _resolve_source(source, claude_code)
    try:
        parsed_fail_on = parse_fail_on(fail_on) if fail_on else None
    except CIConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    with _styled_warnings():
        result = run_ci(
            str(resolved_source),
            baseline_path=str(baseline) if baseline else None,
            pricing_override=str(pricing) if pricing else None,
            quality_file=str(quality) if quality else None,
            fail_on=parsed_fail_on,
            config=cfg,
            update_baseline=update_baseline,
        )

        resolved_pr_comment_out = pr_comment_out or (
            Path(cfg.ci.pr_comment_out) if cfg.ci.pr_comment_out else None
        )
        resolved_sarif_out = sarif_out or (Path(cfg.ci.sarif_out) if cfg.ci.sarif_out else None)
        resolved_junit_out = junit_out or (Path(cfg.ci.junit_out) if cfg.ci.junit_out else None)
        resolved_badge_out = badge_out or (Path(cfg.ci.badge_out) if cfg.ci.badge_out else None)

        if result.report is not None:
            render_terminal(result.report)
            if resolved_pr_comment_out is not None and result.baseline is not None:
                resolved_pr_comment_out.write_text(
                    render_pr_comment(result.report, result.baseline), encoding="utf-8"
                )
            if resolved_sarif_out is not None:
                resolved_sarif_out.write_text(
                    render_sarif(result.report, tool_version=__version__), encoding="utf-8"
                )
            if resolved_junit_out is not None:
                resolved_junit_out.write_text(
                    render_junit(result.report, ci_reasons=result.reasons), encoding="utf-8"
                )
            if resolved_badge_out is not None:
                resolved_badge_out.write_text(render_badge(result.report), encoding="utf-8")
            if report_json_out is not None:
                report_json_out.write_text(render_json(result.report), encoding="utf-8")

    render_ci_verdict(
        result,
        baseline_path=str(baseline) if baseline else cfg.ci.baseline_path,
        baseline_updated=update_baseline,
    )

    raise typer.Exit(code=result.exit_code)


if __name__ == "__main__":
    app()
