"""Core data models (doc §9.1): the normalized shape every adapter maps into
and every detector/scorer/renderer operates on."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    chat = "chat"
    execute_tool = "execute_tool"
    invoke_agent = "invoke_agent"
    create_agent = "create_agent"
    embeddings = "embeddings"
    other = "other"


class QualityRisk(str, Enum):
    none = "none"
    low = "low"
    review = "review"


class Severity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RawSpan(BaseModel):
    """A single OTel span as read off the wire, before normalization.

    Adapters yield these; normalize.py maps them into LLMCall/ToolCall/RetrievalCall.
    """

    span_id: str
    parent_span_id: str | None = None
    trace_id: str | None = None
    name: str
    kind: SpanKind = SpanKind.other
    attributes: dict[str, Any] = Field(default_factory=dict)
    start_ns: int = 0
    end_ns: int = 0
    events: list[dict[str, Any]] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    reasoning: int = 0

    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_creation + self.reasoning

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_creation=self.cache_creation + other.cache_creation,
            reasoning=self.reasoning + other.reasoning,
        )


class Cost(BaseModel):
    """Filled by the Cost Engine; never hand-populated."""

    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_creation: float = 0
    reasoning: float = 0
    total: float = 0
    pricing_version: str = ""
    # True when the model had no registry entry: fields above are left at 0,
    # not a guessed price (doc §9.4 — unknown models warn, never fabricate).
    unpriced: bool = False
    # The registry entry that actually priced this call, when it differs from
    # the call's own model string (a date-suffixed or "-latest" id resolved to
    # its base entry) — provenance for "which rate card produced this number".
    priced_as: str = ""


class LLMCall(BaseModel):
    span_id: str
    parent_id: str | None = None
    provider: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    prompt_fingerprint: str | None = None
    messages: list[dict[str, Any]] | None = None
    # Anthropic's raw API (and Claude Code session logs) break cache_creation
    # down by TTL ({"5m": n, "1h": n}); the 1-hour writes bill at 2x instead
    # of 1.25x. Populated only when the source actually carries the breakdown
    # and it sums to usage.cache_creation — never inferred.
    cache_write_ttl_breakdown: dict[str, int] | None = None
    start_ns: int = 0
    end_ns: int = 0
    cost: Cost = Field(default_factory=Cost)


class ToolCall(BaseModel):
    span_id: str
    parent_id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    args_hash: str = ""
    result: str | None = None
    result_hash: str | None = None
    start_ns: int = 0
    end_ns: int = 0


class RetrievalCall(BaseModel):
    span_id: str
    parent_id: str | None = None
    query: str | None = None
    top_k: int | None = None
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    start_ns: int = 0
    end_ns: int = 0


class Iteration(BaseModel):
    index: int
    llm_calls: list[LLMCall] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    retrievals: list[RetrievalCall] = Field(default_factory=list)

    def tokens(self) -> TokenUsage:
        total = TokenUsage()
        for call in self.llm_calls:
            total = total + call.usage
        return total

    def cost(self) -> float:
        return sum((call.cost.total for call in self.llm_calls), start=0.0)


class Loop(BaseModel):
    loop_id: str
    iterations: list[Iteration] = Field(default_factory=list)
    reached_success: bool = False
    goal_signal: str | None = None
    model_mix: dict[str, int] = Field(default_factory=dict)


class Task(BaseModel):
    task_id: str
    loops: list[Loop] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)


class Session(BaseModel):
    session_id: str
    tasks: list[Task] = Field(default_factory=list)


class Trace(BaseModel):
    source: str
    sessions: list[Session] = Field(default_factory=list)
    # Ingestion-coverage accounting (never silent): spans that carried
    # gen_ai.usage.* token attributes but could not be classified as a call
    # kind, plus duplicate span ids collapsed to their first occurrence, plus
    # spans quarantined for malformed attributes. All surfaced on the Report
    # so an undercount is loud, not invisible.
    dropped_token_spans: int = 0
    duplicate_spans: int = 0
    skipped_spans: list[str] = Field(default_factory=list)


class FindingBasis(str, Enum):
    """How a finding's wasted_dollars figure was derived.

    measured: computed from tokens the provider actually billed (e.g. a
    resent prefix priced at the real rate card) — trustworthy enough to
    gate CI on.
    estimated: a heuristic projection (chars/4 token estimates, policy
    ceilings, hypothetical downgrades) — reported, but excluded from the
    efficiency grade so a guess can never fail a build.
    """

    measured = "measured"
    estimated = "estimated"


class Finding(BaseModel):
    id: str
    subtype: str | None = None
    severity: Severity
    wasted_tokens: int = 0
    wasted_dollars: float = 0.0
    basis: FindingBasis = FindingBasis.measured
    quality_risk: QualityRisk = QualityRisk.none
    evidence: str
    fix: str
    fix_savings_note: str | None = None
    location: str | None = None
    span_ids: list[str] = Field(default_factory=list)


class Score(BaseModel):
    efficiency: int
    grade: str
    waste_ratio: float
    quality_factor: float
    quality_measured: bool
    recoverable_dollars: float
    # Sum of estimated-basis findings (chars/4 projections, policy ceilings,
    # hypothetical downgrades). Kept separate from recoverable_dollars so a
    # heuristic guess is visible but never inflates the measured headline or
    # the grade.
    estimated_extra_dollars: float = 0.0
    monthly_projection: float | None = None


class Report(BaseModel):
    trace_source: str
    # Which adapter read the trace ("otlp", "claude-code") — lets renderers
    # add source-specific context (e.g. Claude Code sessions are priced at
    # standard API rates even for subscription users).
    source_format: str = "otlp"
    total_dollars: float
    token_breakdown: dict[str, int] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    score: Score
    pricing_version: str
    generated_at: str
    # Count of LLM calls the pricing engine couldn't price (unknown model,
    # no override) — `total_dollars` is an undercount whenever this is > 0.
    # Surfaced so CI can fail loudly (doc §11.3 exit code 4) instead of
    # silently gating on a cost figure known to be incomplete.
    unpriced_calls: int = 0
    # The distinct "provider/model" strings behind those unpriced calls
    # (sorted, deduplicated) — so a renderer can say *which* model has no
    # registry entry, not just that something was missed.
    unpriced_models: list[str] = Field(default_factory=list)
    # Ingestion-coverage accounting mirrored from Trace (see Trace): any
    # nonzero value here means total_dollars may be an undercount, and the
    # renderers say so out loud.
    dropped_token_spans: int = 0
    duplicate_spans: int = 0
    skipped_spans: list[str] = Field(default_factory=list)
