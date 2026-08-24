"""Primary adapter: OTLP JSON export (file or stream) -> RawSpans.

Deterministic and offline, which is why the doc makes this the only adapter
required for the Phase 0 exit gate.

Real-world tolerance (v0.2): the OTel GenAI semantic conventions are still
pre-stable and have renamed their load-bearing attributes twice — v1.27
renamed gen_ai.usage.prompt_tokens/completion_tokens to input_tokens/
output_tokens, and v1.37 renamed gen_ai.system to gen_ai.provider.name.
Most deployed instrumentation (OpenLLMetry/Traceloop, OpenLIT, pre-2025 OTel
instrumentations) still emits the older generation, and OpenInference (the
default instrumentation for OpenAI Agents SDK / CrewAI / LangGraph via
Arize) uses its own llm.* attribute namespace entirely. This adapter maps
every known generation onto the canonical current names so downstream code
only ever looks in one place. It also reads both OTLP JSON framings: a
single top-level object, and the spec-standard JSON Lines file
(one TracesData object per line — what the OTel Collector file exporter and
the SDK file exporter actually write), in camelCase or snake_case keys.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import IO, Any

from wattage.adapters.base import Adapter, AdapterError
from wattage.models import RawSpan, SpanKind

# gen_ai.* is the canonical attribute namespace (OTel GenAI semconv). Older
# instrumentation used these names instead; map them onto the canonical key
# so downstream code only ever has to look in one place.
_LEGACY_MODEL_KEYS = ("llm.model", "openai.model", "llm.model_name")

# gen_ai.system is the pre-v1.37 name for gen_ai.provider.name — still what
# most deployed instrumentation emits. OpenInference uses llm.provider (and
# llm.system for the provider family).
_LEGACY_PROVIDER_KEYS = ("gen_ai.system", "llm.provider", "llm.system")

# Pre-v1.27 semconv token names (OpenLLMetry/Traceloop still emit these),
# plus OpenInference's llm.token_count.* namespace.
_LEGACY_INPUT_TOKEN_KEYS = ("gen_ai.usage.prompt_tokens", "llm.token_count.prompt")
_LEGACY_OUTPUT_TOKEN_KEYS = ("gen_ai.usage.completion_tokens", "llm.token_count.completion")

# The official OTel GenAI semconv attribute is "gen_ai.usage.reasoning.
# output_tokens" (registry: opentelemetry.io/docs/specs/semconv/registry/
# attributes/gen-ai/) -- this project's own synthetic fixtures and earlier
# code used the flat "gen_ai.usage.reasoning_tokens" instead, which isn't
# the standardized name. Accepted as a fallback alongside it, as are
# OpenLLMetry's and OpenInference's variants.
_LEGACY_REASONING_KEYS = (
    "gen_ai.usage.reasoning_tokens",
    "llm.token_count.completion_details.reasoning",
)

# Same gap, for cache tokens: the official names are "gen_ai.usage.
# cache_read.input_tokens" / "gen_ai.usage.cache_creation.input_tokens";
# flat variants and OpenInference's prompt_details.* accepted as fallbacks.
_LEGACY_CACHE_READ_KEYS = (
    "gen_ai.usage.cache_read_input_tokens",
    "llm.token_count.prompt_details.cache_read",
)
_LEGACY_CACHE_CREATION_KEYS = (
    "gen_ai.usage.cache_creation_input_tokens",
    "llm.token_count.prompt_details.cache_write",
)

# OpenInference TOOL-span attributes -> the tool names normalize.py reads.
_LEGACY_TOOL_NAME_KEYS = ("tool.name",)
_LEGACY_TOOL_ARGS_KEYS = ("tool.parameters",)

_KIND_BY_OPERATION = {
    "chat": SpanKind.chat,
    # Canonical semconv operations beyond "chat": text_completion is the
    # legacy-completions op, generate_content is what Google GenAI SDK
    # instrumentation emits. Both are LLM inference calls.
    "text_completion": SpanKind.chat,
    "generate_content": SpanKind.chat,
    # "call_llm" isn't a canonical semconv operation name, but it's what
    # mozilla-ai/any-agent (a real, actively maintained multi-framework
    # instrumentation) actually emits — found via real-trace validation, not
    # speculatively added.
    "call_llm": SpanKind.chat,
    "execute_tool": SpanKind.execute_tool,
    "invoke_agent": SpanKind.invoke_agent,
    "create_agent": SpanKind.create_agent,
    "embeddings": SpanKind.embeddings,
}

# OpenInference marks span roles with openinference.span.kind instead of an
# operation name. RETRIEVER spans map onto the embeddings/retrieval kind —
# they're what feeds retrieval analysis.
_KIND_BY_OPENINFERENCE = {
    "LLM": SpanKind.chat,
    "TOOL": SpanKind.execute_tool,
    "AGENT": SpanKind.invoke_agent,
    "RETRIEVER": SpanKind.embeddings,
    "EMBEDDING": SpanKind.embeddings,
}

# Session-identity attributes (span- or resource-level): the semconv
# conversation id and the common session.id convention. Used by sessionize
# to stitch multi-trace conversations back into one session.
_SESSION_KEYS = ("session.id", "gen_ai.conversation.id")


def _decode_value(value: dict[str, Any]) -> Any:
    """Decode one OTLP JSON AnyValue (protobuf JSON mapping: int64 etc. as strings)."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_decode_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return _decode_attributes(value["kvlistValue"].get("values", []))
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _decode_attributes(raw_attrs: list[dict[str, Any]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for kv in raw_attrs:
        key = kv.get("key")
        if key is None:
            continue
        attrs[key] = _decode_value(kv.get("value", {}))
    return attrs


def _first_present(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in attrs:
            return attrs[key]
    return None


def _apply_legacy_aliases(attrs: dict[str, Any]) -> dict[str, Any]:
    # Provenance check BEFORE aliasing: the inclusive-cache marker below
    # must only apply when the input count actually CAME from
    # OpenInference's llm.token_count.prompt. On a dual-emitting span the
    # canonical gen_ai.usage.input_tokens wins the alias — and if that value
    # follows the provider's own (possibly additive) convention, marking it
    # inclusive would subtract cache tokens that were never included,
    # quarantining a well-formed span as "malformed".
    input_from_openinference = (
        "gen_ai.usage.input_tokens" not in attrs and "llm.token_count.prompt" in attrs
    )
    alias_map = {
        "gen_ai.request.model": _LEGACY_MODEL_KEYS,
        "gen_ai.provider.name": _LEGACY_PROVIDER_KEYS,
        "gen_ai.usage.input_tokens": _LEGACY_INPUT_TOKEN_KEYS,
        "gen_ai.usage.output_tokens": _LEGACY_OUTPUT_TOKEN_KEYS,
        "gen_ai.usage.reasoning.output_tokens": _LEGACY_REASONING_KEYS,
        "gen_ai.usage.cache_read.input_tokens": _LEGACY_CACHE_READ_KEYS,
        "gen_ai.usage.cache_creation.input_tokens": _LEGACY_CACHE_CREATION_KEYS,
        "gen_ai.tool.name": _LEGACY_TOOL_NAME_KEYS,
        "gen_ai.tool.call.arguments": _LEGACY_TOOL_ARGS_KEYS,
    }
    for canonical, legacy_keys in alias_map.items():
        if canonical not in attrs:
            value = _first_present(attrs, legacy_keys)
            if value is not None:
                attrs[canonical] = value

    # OpenInference's llm.token_count.prompt is the TOTAL prompt count with
    # prompt_details.* as subsets of it (per the OpenInference spec), i.e.
    # the inclusive convention regardless of provider. Mark it so
    # normalize.py applies the subset split instead of the per-provider
    # default — but only when the input count genuinely came from that
    # attribute (see input_from_openinference above).
    if input_from_openinference and (
        "llm.token_count.prompt_details.cache_read" in attrs
        or "llm.token_count.prompt_details.cache_write" in attrs
    ):
        attrs.setdefault("wattage.cache_tokens_inclusive", True)

    # OpenInference TOOL spans put the tool's output in output.value.
    if "gen_ai.output" not in attrs and "gen_ai.tool.call.result" not in attrs:
        if attrs.get("openinference.span.kind") == "TOOL" and "output.value" in attrs:
            attrs["gen_ai.output"] = attrs["output.value"]
    return attrs


def _classify_kind(attrs: dict[str, Any], span_name: str) -> SpanKind:
    operation = attrs.get("gen_ai.operation.name")
    if isinstance(operation, str) and operation in _KIND_BY_OPERATION:
        return _KIND_BY_OPERATION[operation]
    oi_kind = attrs.get("openinference.span.kind")
    if isinstance(oi_kind, str) and oi_kind.upper() in _KIND_BY_OPENINFERENCE:
        return _KIND_BY_OPENINFERENCE[oi_kind.upper()]
    # Span display names follow "{operation} {model}" (semconv naming rule);
    # fall back to the leading token when gen_ai.operation.name is absent.
    leading = span_name.split(" ", 1)[0]
    return _KIND_BY_OPERATION.get(leading, SpanKind.other)


def _get_either(payload: dict[str, Any], camel: str, snake: str) -> list[Any]:
    """OTLP protobuf-JSON is written in camelCase by the spec, but
    snake_case protobuf-JSON is valid and appears in the wild (collector
    configs, hand-converted traces). Accept both. A value that exists but
    isn't a list is a malformed trace — raised, not silently skipped, so a
    corrupted export can't quietly produce an empty (vacuously passing)
    result."""
    value = payload.get(camel, payload.get(snake))
    if value is None:
        return []
    if not isinstance(value, list):
        raise AdapterError(f"malformed trace: {camel} must be a list, got {type(value).__name__}")
    return value


class OTLPFileAdapter(Adapter):
    format_name = "otlp"

    def supports(self, source: str) -> bool:
        return isinstance(source, str) and source.endswith((".json", ".otlp.json", ".jsonl"))

    def read(self, source: str | IO[str]) -> Iterable[RawSpan]:
        for payload in self._load_payloads(source):
            try:
                for resource_span in _get_either(payload, "resourceSpans", "resource_spans"):
                    resource = resource_span.get("resource", {}) or {}
                    resource_attrs = _decode_attributes(resource.get("attributes", []))
                    session_attrs = {
                        key: resource_attrs[key] for key in _SESSION_KEYS if key in resource_attrs
                    }
                    for scope_span in _get_either(resource_span, "scopeSpans", "scope_spans"):
                        for span in _get_either(scope_span, "spans", "spans"):
                            yield self._to_raw_span(span, session_attrs)
            except AttributeError as exc:
                # resourceSpans/scopeSpans/spans is supposed to be a list of
                # objects; a non-object entry (e.g. a stray string) fails its
                # own .get() call here, one level up from _to_raw_span's per-
                # span validation below.
                raise AdapterError(f"malformed trace: {exc}") from exc

    @staticmethod
    def _load_payloads(source: str | IO[str]) -> list[dict[str, Any]]:
        if isinstance(source, str):
            with open(source, encoding="utf-8") as f:
                text = f.read()
        else:
            text = source.read()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as single_err:
            # Not one JSON document — try the OTLP File Exporter's actual
            # format: JSON Lines, one TracesData object per line (this is
            # what the OTel Collector file exporter writes; the single-
            # object form this adapter originally required is the rarer
            # shape in practice).
            payloads: list[dict[str, Any]] = []
            for line_no, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    raise AdapterError(
                        f"not valid JSON (parse error: {single_err}) and not valid "
                        f"JSON Lines (line {line_no} failed to parse)"
                    ) from single_err
                if not isinstance(parsed, dict):
                    raise AdapterError(
                        f"malformed trace: JSON Lines entry on line {line_no} must be an "
                        f"object, got {type(parsed).__name__}"
                    ) from single_err
                payloads.append(parsed)
            if not payloads:
                raise AdapterError(f"not valid JSON: {single_err}") from single_err
            return payloads
        if not isinstance(raw, dict):
            raise AdapterError(
                f"malformed trace: top-level JSON must be an object, got {type(raw).__name__}"
            )
        return [raw]

    @staticmethod
    def _to_raw_span(span: dict[str, Any], session_attrs: dict[str, Any] | None = None) -> RawSpan:
        # This is the OTLP-JSON/untrusted-input boundary -- a hand-edited or
        # corrupted trace can be malformed in arbitrarily many ways here, and
        # every one of them should read as a bad trace (AdapterError -> `wattage
        # ci` exit 3), not an unhandled traceback masquerading as exit 1 "cost
        # regression detected".
        try:
            span_id = span.get("spanId", span.get("span_id"))
            if not span_id:
                raise KeyError("spanId")
            attrs = _apply_legacy_aliases(_decode_attributes(span.get("attributes", [])))
            # Resource-level session identity applies to every span under the
            # resource, at lower precedence than the span's own attributes.
            for key, value in (session_attrs or {}).items():
                attrs.setdefault(key, value)
            events = [
                {
                    "name": event.get("name", ""),
                    "attributes": _decode_attributes(event.get("attributes", [])),
                }
                for event in span.get("events", [])
            ]
            name = span.get("name", "")
            return RawSpan(
                span_id=span_id,
                parent_span_id=span.get("parentSpanId", span.get("parent_span_id")) or None,
                trace_id=span.get("traceId", span.get("trace_id")),
                name=name,
                kind=_classify_kind(attrs, name),
                attributes=attrs,
                start_ns=int(span.get("startTimeUnixNano", span.get("start_time_unix_nano", 0))),
                end_ns=int(span.get("endTimeUnixNano", span.get("end_time_unix_nano", 0))),
                events=events,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise AdapterError(f"malformed span in trace: {exc}") from exc
