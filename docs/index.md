![wattage](assets/logo.svg#only-light){ height="56" }
![wattage](assets/logo-dark.svg#only-dark){ height="56" }

[![CI](https://github.com/faizannraza/wattage/actions/workflows/ci.yml/badge.svg)](https://github.com/faizannraza/wattage/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/wattage.svg)](https://pypi.org/project/wattage/)
[![npm](https://img.shields.io/npm/v/wattage-cli.svg)](https://www.npmjs.com/package/wattage-cli)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/faizannraza/wattage/blob/main/LICENSE)

**Find the tokens your AI agent wasted — in dollars, with the fix — and fail
the PR when a change makes your agent more expensive.**

Wattage reads the traces and session logs your agents already produce — local
**Claude Code sessions**, or any **[OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)**
trace export — prices every call against a verified, dated pricing snapshot,
runs ten waste-pattern detectors that each name a dollar figure and a
concrete fix, computes a quality-aware Token Efficiency score, and can fail
your CI when a change makes your agents more expensive. Fully offline, no
API key, nothing phones home.

## Quick start

```bash
uvx wattage demo                    # findings-rich sample report, zero setup
uvx wattage report --claude-code    # your latest Claude Code session
uvx wattage report trace.json       # any OTLP GenAI trace export
```

The input format is auto-detected. Don't have a trace? Your Claude Code
sessions under `~/.claude/projects` already count — see
[Getting your first trace](getting-started.md) for every path from zero.

## What it is (and isn't)

Wattage is a **diagnosis + prescription + gate**, not another dashboard:

- It **consumes** the traces your existing observability tool (Langfuse, Helicone, an OTel Collector, whatever) already produces — it doesn't replace them.
- It **names specific waste patterns** with dollar figures and a fix, rather than just showing you a bill.
- It **never enforces anything automatically** — findings are recommendations you apply, not prompts it silently rewrites. Runtime enforcement (killing waste in-flight) is a deliberately separate, later, opt-in capability.

## The four surfaces

1. **`wattage report`** — a priced, findings-quantified report for one trace: terminal, JSON, or a self-contained HTML **burn map** (an interactive flame graph whose findings light up the exact frames that burned the money).
2. **`wattage score` / `wattage badge`** — a single 0–100 Token Efficiency grade you can drop into a README.
3. **`wattage ci`** — a cost-regression gate: fails a PR when a change makes your agent measurably more expensive, with exact, documented exit codes and an unmissable PASS/FAIL verdict.
4. **`wattage demo`** — a bundled findings-rich trace, so the first command you ever run shows the product doing its job.

## Where to go next

- **[Getting your first trace](getting-started.md)** — Claude Code sessions, existing OTel traces, or instrumenting from zero.
- **[Adapters](adapters.md)** — every format Wattage reads (Claude Code JSONL, all OTel GenAI attribute generations, OpenInference, JSON Lines), and exactly how far each has been validated.
- **[Detectors](detectors/index.md)** — the ten waste patterns Wattage looks for, one page each, including the measured-vs-estimated basis rule.
- **[The Convergence Engine](convergence.md)** — how Wattage catches agents thrashing in unproductive loops, why exact-match duplicate detection can't, and what the benchmark does and does not show.
- **[CI Integration](ci.md)** — wiring the GitHub Action so cost regressions fail the build.
- **[Configuration](configuration.md)** — every threshold, with defaults.

## A note on honesty

Every dollar figure, every score, and every "typical savings" claim Wattage produces comes from an actual computation against your real trace and the current vendored pricing data — never a guess dressed up as a number. Findings whose dollars are heuristic projections are labeled `estimated`, printed with a `~`, and can never move the grade or fail a CI build. When Wattage doesn't have enough information to say something confidently (an unpriced model, an unmeasured quality signal, a trace with no captured usage, a genuinely ambiguous loop), it says so explicitly instead of filling the gap with a plausible-looking number. That's a design principle, not an afterthought — see the individual detector pages for exactly which limitations each one is honest about.
