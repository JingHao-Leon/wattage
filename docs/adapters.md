# Adapters

An adapter turns a trace source into the stream of spans Wattage normalizes
into calls, iterations, loops, tasks, and sessions. Two adapters ship today
— **OTLP JSON** (the general OpenTelemetry path) and **Claude Code
sessions** (the zero-instrumentation path) — and the right one is picked by
looking at the file's *content*, never its extension: `wattage report
<file>` just works.

Honesty note on validation levels: entries below say explicitly whether
they were validated against a **real captured trace** or are
**attribute-mapping support** built from the format's published spec. Both
are tested against fixtures; only the former has met real-world data.

## Claude Code session adapter

Reads the session `.jsonl` files Claude Code writes under
`~/.claude/projects/<project>/<session-id>.jsonl` — validated against real
sessions. Point Wattage at a session file, a project directory, or nothing
at all:

```bash
wattage report --claude-code            # newest session across all projects
wattage report <session>.jsonl          # one session
wattage report <project-directory>/     # every session in the project
```

What it handles, because real session files require it:

- **One API response, many records.** Claude Code writes one JSONL record
  per content block, each repeating the same message id and usage object —
  the adapter aggregates by message id, so a four-block response is billed
  once, not four times.
- **Reasoning tokens** (`output_tokens_details.thinking_tokens`) split out
  of output, exactly like the OTel path.
- **The cache-write TTL breakdown.** Anthropic bills 1-hour-TTL cache
  writes at 2x base input (5-minute writes at 1.25x). Session logs carry
  the 5m/1h split (`cache_creation.ephemeral_*`), so Wattage prices each
  at its real rate — a distinction OTel `gen_ai.*` attributes can't even
  express. The breakdown is only used when it sums to the reported
  cache-creation total; inconsistent data falls back to the safe 5-minute
  rate.
- **Tool activity.** `tool_use` blocks become tool calls; their
  `tool_result` records are matched back by id, feeding the convergence,
  redundancy, and bloat detectors.
- **Sub-agents.** Sidechain records (`isSidechain`) are grouped into their
  own task per sidechain, so a sub-agent's loop is analyzed on its own
  rather than interleaved into the main conversation.
- **Synthetic records** (`"<synthetic>"` model — no API call, no cost) and
  session bookkeeping records are skipped.

Costs are standard Anthropic API rates; for subscription users that's the
API-equivalent value of the session, and the report says so.

## OTLP file adapter

Point any command at an [OTLP JSON](https://opentelemetry.io/docs/specs/otlp/)
trace export. Both real-world framings are read:

- a single top-level `resourceSpans[].scopeSpans[].spans[]` object, and
- spec-standard **JSON Lines** (one TracesData object per line — what the
  OTel Collector file exporter and the SDK file exporter actually write),

in camelCase or snake_case protobuf-JSON keys.

### Every attribute generation, not just the newest

The GenAI semantic conventions are still pre-stable and have renamed their
load-bearing attributes twice: v1.27 renamed
`gen_ai.usage.prompt_tokens`/`completion_tokens` to
`input_tokens`/`output_tokens`, and v1.37 renamed `gen_ai.system` to
`gen_ai.provider.name`. Most deployed instrumentation still emits the older
generation. Wattage reads them all and maps everything onto the canonical
names:

| Wattage's canonical name | Also accepted |
|---|---|
| `gen_ai.provider.name` | `gen_ai.system`, `llm.provider`, `llm.system` |
| `gen_ai.request.model` | `llm.model`, `openai.model`, `llm.model_name` |
| `gen_ai.usage.input_tokens` | `gen_ai.usage.prompt_tokens`, `llm.token_count.prompt` |
| `gen_ai.usage.output_tokens` | `gen_ai.usage.completion_tokens`, `llm.token_count.completion` |
| `gen_ai.usage.reasoning.output_tokens` | `gen_ai.usage.reasoning_tokens`, `llm.token_count.completion_details.reasoning` |
| `gen_ai.usage.cache_read.input_tokens` | `gen_ai.usage.cache_read_input_tokens`, `llm.token_count.prompt_details.cache_read` |
| `gen_ai.usage.cache_creation.input_tokens` | `gen_ai.usage.cache_creation_input_tokens`, `llm.token_count.prompt_details.cache_write` |

Operation names: `chat`, `text_completion`, `generate_content`, `call_llm`
(any-agent's name — found via real-trace validation), `execute_tool`,
`invoke_agent`, `create_agent`, `embeddings`, plus OpenInference's
`openinference.span.kind` (`LLM`/`TOOL`/`AGENT`/`RETRIEVER`/`EMBEDDING`).

### OpenInference (attribute-mapping support)

[OpenInference](https://github.com/Arize-ai/openinference) is the default
instrumentation path for the OpenAI Agents SDK, CrewAI, and LangGraph via
Arize Phoenix. Its `llm.token_count.*` names map one-to-one onto Wattage's
model (see the table above), including the cache and reasoning details, and
its `prompt` count is treated as inclusive of cache tokens — per the
OpenInference spec, which defines `prompt_details.*` as subsets of the
prompt total. This support is spec-derived and fixture-tested; if you run a
real OpenInference trace through Wattage, an issue report (works cleanly /
tripped on X) is genuinely valuable.

### Validated against a real trace

Wattage's real-trace validation (against a genuine
[mozilla-ai/any-agent](https://github.com/mozilla-ai/any-agent) trace, not
a synthetic fixture) surfaced four concrete gaps, all now handled: the
`call_llm` operation alias, litellm-style `"provider/model"` strings
(Wattage splits on the first `/` when no explicit provider is given),
legacy model attribute names, and the tool-call attribute variants
(`gen_ai.tool.args` / generic `gen_ai.output`). None were added
speculatively — each was found by running Wattage against real data and
fixing what broke. See `benchmarks/traces/README.md` for the trace's full
provenance.

### Session stitching

Production agents emit one OTel trace per user turn, which would shatter a
conversation into unrelated sessions. Wattage reads `session.id` /
`gen_ai.conversation.id` (span- or resource-level) and stitches multi-trace
conversations back into one session before analysis.

### Coverage accounting — an undercount is never silent

Spans that carry `gen_ai.usage.*` token attributes but no recognizable call
kind, duplicate span ids (OTLP delivery is at-least-once), and quarantined
malformed spans are **counted and reported** on every surface — a trace
whose totals may be an undercount says so out loud. A trace with spans but
zero captured usage refuses to grade entirely rather than printing a
vacuous A (100).

## Pricing

Wattage never fabricates a price. It ships a vendored, versioned pricing
snapshot (`src/wattage/pricing/data/pricing.yaml`) — 52 models across
anthropic, openai, google, mistral, and xai, every number sourced from the
provider's own pricing page with the fetch date recorded. The registry is:

- **Context-tier aware** — Gemini and Grok reprice the whole request once
  the prompt crosses 200k tokens; Wattage selects the tier from each call's
  own prompt size. A published-but-rateless range (OpenAI's rates above
  272K context aren't on its pricing page) is left **unpriced**, not billed
  at the wrong tier.
- **Effective-date aware** — rates with a published validity period
  (Gemini 3.7/3.6 Flash promo pricing through 2026-12-31) price by the
  call's own timestamp.
- **Cached-input exact** — absolute cached-input rates where providers
  publish them (they're not a uniform 0.1x: gpt-4o's cached rate is 0.5x
  input, Grok's range 0.15–0.25x).
- **Model-id tolerant, carefully** — Anthropic-style compact date
  suffixes (`claude-sonnet-4-5-20250929`) and `-latest` tags resolve to
  their base entry (Anthropic documents the dateless id as an alias of the
  dated snapshot), with the resolution recorded on the call's cost for
  provenance. OpenAI's dashed dated SKUs (`gpt-4o-2024-05-13`) deliberately
  do **not** resolve — they are genuinely different price points, so they
  stay unpriced rather than billed at the wrong snapshot's rate — and
  Bedrock/Vertex ids are never silently priced at first-party rates:
  partner pricing genuinely differs, so those need an explicit override.

If a call uses a model with no registry entry, Wattage leaves that call's
cost at zero rather than guessing a rate — and every surface says so
plainly rather than showing a plausible-looking grade on incomplete data:
`wattage report`/`score`/`badge` name the unpriced model and mark the total
cost and score as incomplete (never a quiet "A (100)"), and `wattage ci`
treats any unpriced call as a hard pricing error (exit code 4), because a
cost-regression gate built on an undercount isn't trustworthy.

You can override or extend the registry with your own `pricing.yaml` via
`--pricing`, useful for negotiated enterprise rates, self-hosted models, or
providers not yet vendored (DeepSeek's time-of-day rates don't fit the
schema yet, so it's excluded rather than mispriced). Overrides merge
**field-level**: overriding just `input` keeps the vendored cache
multipliers instead of silently resetting them.

## What's next

A Langfuse export adapter, a live OTLP endpoint/collector mode, and
additional coding-agent session formats are on the roadmap — the normalized
data model (sessions → tasks → loops → iterations → calls) doesn't change;
only the ingestion layer does.
