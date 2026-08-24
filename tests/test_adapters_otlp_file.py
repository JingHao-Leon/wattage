import json
from pathlib import Path

import pytest

from wattage.adapters.base import AdapterError
from wattage.adapters.otlp_file import OTLPFileAdapter
from wattage.models import SpanKind
from wattage.normalize import normalize_llm_call
from wattage.report import build_report


def _write_otlp(tmp_path: Path, spans: list[dict]) -> Path:
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))
    return path


def test_supports_only_json_paths() -> None:
    adapter = OTLPFileAdapter()
    assert adapter.supports("trace.json") is True
    assert adapter.supports("trace.otlp.json") is True
    assert adapter.supports("trace.pb") is False


def test_reads_canonical_gen_ai_attributes(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s1",
                "parentSpanId": "",
                "name": "chat claude-sonnet-4-6",
                "startTimeUnixNano": "1000",
                "endTimeUnixNano": "2000",
                "attributes": [
                    {"key": "gen_ai.provider.name", "value": {"stringValue": "anthropic"}},
                    {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-6"}},
                    {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "18450"}},
                    {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "320"}},
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert len(spans) == 1
    span = spans[0]
    assert span.span_id == "s1"
    assert span.parent_span_id is None
    assert span.kind == SpanKind.chat
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert span.attributes["gen_ai.usage.input_tokens"] == 18450
    assert span.start_ns == 1000
    assert span.end_ns == 2000


def test_legacy_model_attribute_names_map_to_canonical(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s-legacy-llm",
                "name": "chat",
                "attributes": [{"key": "llm.model", "value": {"stringValue": "claude-legacy"}}],
            },
            {
                "traceId": "t1",
                "spanId": "s-legacy-openai",
                "name": "chat",
                "attributes": [{"key": "openai.model", "value": {"stringValue": "gpt-4o-legacy"}}],
            },
        ],
    )
    spans = {s.span_id: s for s in OTLPFileAdapter().read(str(path))}
    assert spans["s-legacy-llm"].attributes["gen_ai.request.model"] == "claude-legacy"
    assert spans["s-legacy-openai"].attributes["gen_ai.request.model"] == "gpt-4o-legacy"


def test_legacy_flat_reasoning_tokens_attribute_maps_to_canonical_name(tmp_path: Path) -> None:
    """gen_ai.usage.reasoning_tokens (flat) is what this project's own
    synthetic fixtures used before the official OTel GenAI semconv
    registry name (gen_ai.usage.reasoning.output_tokens) was confirmed --
    accepted as a fallback so real instrumentation using either name
    still works."""
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "chat",
                "attributes": [
                    {"key": "gen_ai.usage.reasoning_tokens", "value": {"intValue": "42"}}
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert spans[0].attributes["gen_ai.usage.reasoning.output_tokens"] == 42


def test_canonical_reasoning_tokens_attribute_wins_over_legacy(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "chat",
                "attributes": [
                    {"key": "gen_ai.usage.reasoning_tokens", "value": {"intValue": "999"}},
                    {"key": "gen_ai.usage.reasoning.output_tokens", "value": {"intValue": "42"}},
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert spans[0].attributes["gen_ai.usage.reasoning.output_tokens"] == 42


def test_legacy_flat_cache_token_attributes_map_to_canonical_names(tmp_path: Path) -> None:
    """Same gap as reasoning tokens, for cache tokens: the official OTel
    names are gen_ai.usage.cache_read.input_tokens /
    cache_creation.input_tokens (dotted); this project's own code used the
    flat gen_ai.usage.cache_read_input_tokens / cache_creation_input_tokens
    instead. Accepted as fallbacks alongside the canonical names."""
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "chat",
                "attributes": [
                    {"key": "gen_ai.usage.cache_read_input_tokens", "value": {"intValue": "10"}},
                    {
                        "key": "gen_ai.usage.cache_creation_input_tokens",
                        "value": {"intValue": "20"},
                    },
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert spans[0].attributes["gen_ai.usage.cache_read.input_tokens"] == 10
    assert spans[0].attributes["gen_ai.usage.cache_creation.input_tokens"] == 20


def test_canonical_model_attribute_wins_over_legacy(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s1",
                "name": "chat",
                "attributes": [
                    {"key": "llm.model", "value": {"stringValue": "should-not-be-used"}},
                    {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-6"}},
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert spans[0].attributes["gen_ai.request.model"] == "claude-sonnet-4-6"


def test_classifies_kind_from_span_name_when_operation_attribute_absent(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [{"traceId": "t1", "spanId": "s1", "name": "execute_tool search_docs", "attributes": []}],
    )
    spans = list(OTLPFileAdapter().read(str(path)))
    assert spans[0].kind == SpanKind.execute_tool


def test_span_missing_span_id_raises_adapter_error_not_a_bare_key_error(tmp_path: Path) -> None:
    """The real bug this closes: a span missing 'spanId' used to raise a
    bare KeyError straight out of the adapter, which `wattage ci` didn't
    catch -- it fell through as an uncaught exception (Python's default
    exit code 1), indistinguishable from exit code 1's real meaning ("a
    fail-on threshold breached"), instead of the documented exit code 3
    ("ingestion error"). AdapterError is what ci.py now catches."""
    path = _write_otlp(tmp_path, [{"traceId": "t1", "name": "chat", "attributes": []}])
    with pytest.raises(AdapterError, match="spanId"):
        list(OTLPFileAdapter().read(str(path)))


def test_non_dict_span_entry_raises_adapter_error(tmp_path: Path) -> None:
    payload = {"resourceSpans": [{"scopeSpans": [{"spans": "not-a-list-of-objects"}]}]}
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(AdapterError):
        list(OTLPFileAdapter().read(str(path)))


def test_non_object_top_level_json_raises_adapter_error(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(AdapterError, match="top-level JSON must be an object"):
        list(OTLPFileAdapter().read(str(path)))


def _legacy_semconv_span() -> dict:
    """Pre-v1.37/pre-v1.27 semconv generation: gen_ai.system instead of
    gen_ai.provider.name, prompt_tokens/completion_tokens instead of
    input_tokens/output_tokens -- still what most deployed instrumentation
    (OpenLLMetry/Traceloop, OpenLIT, pre-2025 OTel) emits."""
    return {
        "traceId": "t1",
        "spanId": "s1",
        "name": "chat claude-sonnet-4-6",
        "startTimeUnixNano": "1000",
        "endTimeUnixNano": "2000",
        "attributes": [
            {"key": "gen_ai.system", "value": {"stringValue": "anthropic"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-6"}},
            {"key": "gen_ai.usage.prompt_tokens", "value": {"intValue": "100"}},
            {"key": "gen_ai.usage.completion_tokens", "value": {"intValue": "10"}},
        ],
    }


def test_legacy_semconv_generation_aliases_onto_canonical_names(tmp_path: Path) -> None:
    path = _write_otlp(tmp_path, [_legacy_semconv_span()])
    spans = list(OTLPFileAdapter().read(str(path)))

    attrs = spans[0].attributes
    assert attrs["gen_ai.provider.name"] == "anthropic"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 10


def test_legacy_semconv_generation_prices_end_to_end(tmp_path: Path) -> None:
    """The aliasing has to survive the whole pipeline, not just the adapter:
    a legacy-generation trace must come out priced at the model's real rate
    card, not silently unpriced because the provider name lived under the
    old key."""
    path = _write_otlp(tmp_path, [_legacy_semconv_span()])
    report = build_report(str(path))

    assert report.unpriced_calls == 0
    # anthropic/claude-sonnet-4-6: input 3.0e-6, output 15.0e-6.
    assert report.total_dollars == pytest.approx(100 * 3.0e-6 + 10 * 15.0e-6)


def test_openinference_llm_span_subtracts_cache_from_inclusive_prompt_total(
    tmp_path: Path,
) -> None:
    """OpenInference's llm.token_count.prompt is spec-defined as the TOTAL
    prompt count with prompt_details.* as subsets of it -- so the adapter
    must mark the span inclusive, and normalize must subtract cache_read
    from input. Treating the fields as additive (Anthropic's convention)
    would double-bill every cached token."""
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s-oi-llm",
                "name": "ChatCompletion",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
                    {"key": "llm.provider", "value": {"stringValue": "anthropic"}},
                    {"key": "llm.model_name", "value": {"stringValue": "claude-sonnet-4-6"}},
                    {"key": "llm.token_count.prompt", "value": {"intValue": "1000"}},
                    {"key": "llm.token_count.completion", "value": {"intValue": "100"}},
                    {
                        "key": "llm.token_count.prompt_details.cache_read",
                        "value": {"intValue": "400"},
                    },
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))

    span = spans[0]
    assert span.kind == SpanKind.chat
    assert span.attributes["gen_ai.request.model"] == "claude-sonnet-4-6"
    assert span.attributes["gen_ai.usage.input_tokens"] == 1000
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == 400
    assert span.attributes["wattage.cache_tokens_inclusive"] is True

    call = normalize_llm_call(span)
    assert call.usage.input == 600  # 1000 total prompt minus the 400 cached subset
    assert call.usage.cache_read == 400
    assert call.usage.output == 100


def test_openinference_tool_span_classifies_as_execute_tool(tmp_path: Path) -> None:
    path = _write_otlp(
        tmp_path,
        [
            {
                "traceId": "t1",
                "spanId": "s-oi-tool",
                "name": "search",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "TOOL"}},
                    {"key": "tool.name", "value": {"stringValue": "search"}},
                    {"key": "output.value", "value": {"stringValue": "three results"}},
                ],
            }
        ],
    )
    spans = list(OTLPFileAdapter().read(str(path)))

    span = spans[0]
    assert span.kind == SpanKind.execute_tool
    assert span.attributes["gen_ai.tool.name"] == "search"
    assert span.attributes["gen_ai.output"] == "three results"


def test_json_lines_file_yields_spans_from_every_line(tmp_path: Path) -> None:
    """The OTel Collector file exporter writes JSON Lines (one TracesData
    object per line), not a single JSON document -- reading only the first
    line would silently drop every later batch's spans from the bill."""

    def _line(span_id: str) -> str:
        payload = {
            "resourceSpans": [
                {"scopeSpans": [{"spans": [{"traceId": "t1", "spanId": span_id, "name": "chat"}]}]}
            ]
        }
        return json.dumps(payload)

    path = tmp_path / "trace.jsonl"
    path.write_text(_line("line1-span") + "\n" + _line("line2-span") + "\n")

    assert OTLPFileAdapter().supports(str(path)) is True
    spans = list(OTLPFileAdapter().read(str(path)))
    assert [s.span_id for s in spans] == ["line1-span", "line2-span"]


def test_snake_case_otlp_keys_parse_equivalently_to_camel_case(tmp_path: Path) -> None:
    """snake_case protobuf-JSON is valid OTLP and appears in the wild
    (collector configs, hand-converted traces) -- it must parse to the same
    RawSpan as the spec's camelCase form."""
    payload = {
        "resource_spans": [
            {
                "scope_spans": [
                    {
                        "spans": [
                            {
                                "trace_id": "t-snake",
                                "span_id": "s-snake",
                                "name": "chat claude-sonnet-4-6",
                                "start_time_unix_nano": "1000",
                                "end_time_unix_nano": "2000",
                                "attributes": [
                                    {
                                        "key": "gen_ai.usage.input_tokens",
                                        "value": {"intValue": "7"},
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))

    spans = list(OTLPFileAdapter().read(str(path)))
    assert len(spans) == 1
    span = spans[0]
    assert span.span_id == "s-snake"
    assert span.trace_id == "t-snake"
    assert span.kind == SpanKind.chat
    assert span.start_ns == 1000
    assert span.end_ns == 2000
    assert span.attributes["gen_ai.usage.input_tokens"] == 7


def test_string_valued_resource_spans_raises_adapter_error(tmp_path: Path) -> None:
    """A *Spans value that exists but isn't a list is a corrupted export --
    it must raise (AdapterError -> `wattage ci` exit 3), not be silently
    skipped into an empty, vacuously passing result."""
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"resourceSpans": "not-a-list"}))

    with pytest.raises(AdapterError, match="resourceSpans must be a list"):
        list(OTLPFileAdapter().read(str(path)))
