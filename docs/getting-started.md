# Getting your first trace

Wattage auto-detects its input format. Which path below applies depends on
what you already have — and if you use Claude Code, you already have
everything you need.

## You use Claude Code

Every Claude Code session is already logged as a JSONL file under
`~/.claude/projects` — real API calls, real token counts (including the
cache-write TTL breakdown), real tool activity. Nothing to instrument:

```bash
uvx wattage report --claude-code           # your most recent session
uvx wattage report path/to/session.jsonl   # a specific session file
uvx wattage report ~/.claude/projects/<project-dir>/   # every session in a project
```

Costs are computed at standard Anthropic API rates. If you're on a
subscription plan that's the API-equivalent value of the session (the
report says so) — still exactly the number you want for spotting waste
patterns and comparing sessions.

## You already have OTel GenAI traces

If your agent framework or observability tool (an OTel Collector,
OpenLLMetry, OpenInference/Arize instrumentation, Pydantic AI/Logfire, or
anything else that speaks the GenAI semantic conventions) exports OTLP
JSON — single-object or JSON Lines — point Wattage at the export directly:

```bash
uvx wattage report your_trace.json    # .jsonl works too
```

Check your tool's docs for an "OTLP export," "OTel Collector integration,"
or "file exporter" option — see [Adapters](adapters.md) for exactly which
attribute generations and operation-name variants Wattage tolerates (all of
them, in practice: the semconv has renamed its load-bearing attributes
twice, and Wattage reads every generation).

## You have zero instrumentation

Wrap one real LLM call in a real OpenTelemetry span, then export it. This is
the actual minimum — three attributes and the two token counts your
provider's SDK already returns with its response:

```python
# examples/instrument_minimal.py in this repo has the full, runnable version
with tracer.start_as_current_span("chat claude-sonnet-4-6") as span:
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", "anthropic")
    span.set_attribute("gen_ai.request.model", "claude-sonnet-4-6")
    # ... your real API call here ...
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
```

Run the complete version end to end:

```bash
pip install wattage opentelemetry-sdk opentelemetry-exporter-otlp-proto-common
python examples/instrument_minimal.py   # writes trace.json
wattage report trace.json
```

That produces a real `trace.json` — encoded via OpenTelemetry's own official
OTLP JSON encoder, not an approximation of the format — and a real, correctly
priced report. From here, replace the stand-in call in the script with your
actual agent's LLM calls (wrap each one in its own span), add a `tool` span
with `gen_ai.operation.name: "execute_tool"` for any tool calls, and you
have a real trace of your own agent for Wattage's detectors — including the
convergence engine, which needs multiple iterations of tool activity to have
anything to analyze.

## Just want to see it work first?

No clone, no trace, no setup — a findings-rich demo trace ships inside the
package:

```bash
uvx wattage demo
```
