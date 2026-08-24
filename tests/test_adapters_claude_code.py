"""Claude Code session adapter tests: ~/.claude/projects JSONL -> RawSpans.

Fixtures reproduce the real session-record shapes (verified against actual
~/.claude/projects files), including the format's one dangerous quirk: a
single API response is split across MULTIPLE consecutive JSONL records (one
per content block), each repeating the same message.id and the same usage
object. Counting each record would overbill a response by up to the number
of its content blocks.
"""

import json
import os
from pathlib import Path

import pytest

from wattage.adapters import ClaudeCodeAdapter, OTLPFileAdapter, resolve_adapter
from wattage.adapters.claude_code import find_latest_session
from wattage.models import SpanKind
from wattage.report import build_report, build_trace_and_report

_USAGE = {
    "input_tokens": 2,
    "output_tokens": 92,
    "cache_read_input_tokens": 26386,
    "cache_creation_input_tokens": 12034,
    "output_tokens_details": {"thinking_tokens": 40},
    "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 12034},
}


def _assistant(
    message_id: str,
    uuid: str,
    content: list[dict] | None = None,
    parent_uuid: str | None = None,
    model: str = "claude-fable-5",
    usage: dict | None = None,
    timestamp: str = "2026-08-23T10:00:00.000Z",
    sidechain: bool = False,
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "sessionId": "sess-1",
        "timestamp": timestamp,
        "isSidechain": sidechain,
        "requestId": "req_1",
        "message": {
            "id": message_id,
            "model": model,
            "content": content if content is not None else [{"type": "text", "text": "..."}],
            "usage": usage if usage is not None else dict(_USAGE),
        },
    }


def _tool_result(
    tool_use_id: str,
    content: str,
    uuid: str = "u-result",
    timestamp: str = "2026-08-23T10:00:02.000Z",
) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "sessionId": "sess-1",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
        },
    }


def _write_session(tmp_path: Path, records: list[dict], name: str = "sess-1.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_duplicate_message_id_records_count_usage_once(tmp_path: Path) -> None:
    """One API response split across three JSONL records (one per content
    block, same message.id, same usage) must yield exactly ONE chat span
    with that usage counted once — counting each record would overbill the
    response 3x, the exact plausible-looking-but-wrong number this adapter
    exists to prevent."""
    records = [
        _assistant("msg_01", "u1", content=[{"type": "thinking", "thinking": "hm"}]),
        _assistant("msg_01", "u2", content=[{"type": "text", "text": "part one"}]),
        _assistant("msg_01", "u3", content=[{"type": "text", "text": "part two"}]),
    ]
    path = _write_session(tmp_path, records)

    spans = list(ClaudeCodeAdapter().read(str(path)))
    chats = [s for s in spans if s.kind == SpanKind.chat]

    assert len(chats) == 1
    chat = chats[0]
    assert chat.span_id == "msg_01"
    assert chat.attributes["gen_ai.provider.name"] == "anthropic"
    assert chat.attributes["gen_ai.request.model"] == "claude-fable-5"
    assert chat.attributes["gen_ai.usage.input_tokens"] == 2
    assert chat.attributes["gen_ai.usage.output_tokens"] == 92
    assert chat.attributes["gen_ai.usage.cache_read.input_tokens"] == 26386
    assert chat.attributes["gen_ai.usage.cache_creation.input_tokens"] == 12034
    assert chat.attributes["gen_ai.usage.reasoning.output_tokens"] == 40
    assert chat.attributes["gen_ai.usage.cache_creation.ephemeral_5m_input_tokens"] == 0
    assert chat.attributes["gen_ai.usage.cache_creation.ephemeral_1h_input_tokens"] == 12034


def test_tool_use_blocks_become_execute_tool_spans_with_matched_result(tmp_path: Path) -> None:
    """A tool_use content block plus its later user-record tool_result must
    come out as one execute_tool span carrying name, arguments, and the
    matched result text — the inputs the redundancy/bloat detectors read."""
    tool_block = {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "Read",
        "input": {"file_path": "/x"},
    }
    records = [
        _assistant("msg_01", "u1", content=[tool_block]),
        _tool_result("toolu_01", "file contents here"),
    ]
    path = _write_session(tmp_path, records)

    spans = list(ClaudeCodeAdapter().read(str(path)))
    tools = [s for s in spans if s.kind == SpanKind.execute_tool]

    assert len(tools) == 1
    tool = tools[0]
    assert tool.span_id == "toolu_01"
    assert tool.attributes["gen_ai.tool.name"] == "Read"
    assert json.loads(tool.attributes["gen_ai.tool.call.arguments"]) == {"file_path": "/x"}
    assert tool.attributes["gen_ai.tool.call.result"] == "file contents here"


def test_noise_records_and_synthetic_model_are_skipped(tmp_path: Path) -> None:
    """mode/file-history-snapshot/attachment records are session bookkeeping
    with no cost dimension, and "<synthetic>" assistant records are locally
    generated (no API call, no cost) — none of them may produce a span."""
    records = [
        {"type": "mode", "sessionId": "sess-1", "mode": "default"},
        {"type": "file-history-snapshot", "messageId": "fh1", "snapshot": {}},
        {"type": "attachment", "sessionId": "sess-1", "attachment": {"type": "diagnostics"}},
        _assistant("msg_synth", "u0", model="<synthetic>"),
        _assistant("msg_01", "u1"),
    ]
    path = _write_session(tmp_path, records)

    spans = list(ClaudeCodeAdapter().read(str(path)))

    chat_ids = {s.span_id for s in spans if s.kind == SpanKind.chat}
    assert chat_ids == {"msg_01"}
    # Nothing but the synthetic agent root and the one real chat span.
    assert {s.kind for s in spans} == {SpanKind.invoke_agent, SpanKind.chat}


def test_build_report_prices_a_session_end_to_end(tmp_path: Path) -> None:
    """Full pipeline over a split-record session: source format, token
    breakdown (reasoning split out of output, cache classes separate), and
    dollars at pricing.yaml's anthropic/claude-fable-5 rates — with the
    all-1h cache write billed at cache_write_mult_1h (2x base input), not
    the default 5m 1.25x."""
    records = [
        _assistant("msg_01", "u1", content=[{"type": "text", "text": "part one"}]),
        _assistant("msg_01", "u2", content=[{"type": "text", "text": "part two"}]),
        _assistant("msg_01", "u3", content=[{"type": "text", "text": "part three"}]),
        {"type": "mode", "sessionId": "sess-1", "mode": "default"},
    ]
    path = _write_session(tmp_path, records)

    report = build_report(str(path))

    assert report.source_format == "claude-code"
    assert report.unpriced_calls == 0
    # normalize.py splits thinking out of the provider's raw output count:
    # visible output = 92 - 40, reasoning tracked separately.
    assert report.token_breakdown == {
        "input": 2,
        "output": 52,
        "cache_read": 26386,
        "cache_creation": 12034,
        "reasoning": 40,
    }
    # fable-5 rates: input 10.0e-6, output 50.0e-6, cache_read_mult 0.10,
    # cache_write_mult_1h 2.0 (the TTL breakdown says every write is 1h).
    expected = (
        2 * 10.0e-6  # input
        + 52 * 50.0e-6  # visible output
        + 40 * 50.0e-6  # reasoning, billed at the output rate
        + 26386 * 10.0e-6 * 0.10  # cache read
        + 12034 * 10.0e-6 * 2.0  # 1h-TTL cache write at 2x base input
    )
    assert report.total_dollars == pytest.approx(expected)


def test_sidechain_records_land_in_their_own_task(tmp_path: Path) -> None:
    """Sub-agent activity (isSidechain: true) must become its own task per
    sidechain root, so the sub-agent's loop is analyzed on its own instead
    of interleaved into the main conversation's iteration sequence."""
    records = [
        _assistant("msg_main", "u1", timestamp="2026-08-23T10:00:00.000Z"),
        # The sidechain's own parentUuid chain: a sidechain user record is
        # the root, with the sidechain assistant records chained under it.
        {
            "type": "user",
            "uuid": "s0",
            "parentUuid": None,
            "sessionId": "sess-1",
            "isSidechain": True,
            "timestamp": "2026-08-23T10:00:05.000Z",
            "message": {"role": "user", "content": "sub-agent prompt"},
        },
        _assistant(
            "msg_side1",
            "s1",
            parent_uuid="s0",
            sidechain=True,
            timestamp="2026-08-23T10:00:06.000Z",
        ),
        _assistant(
            "msg_side2",
            "s2",
            parent_uuid="s1",
            sidechain=True,
            timestamp="2026-08-23T10:00:07.000Z",
        ),
    ]
    path = _write_session(tmp_path, records)

    trace, _report = build_trace_and_report(str(path))

    assert len(trace.sessions) == 1
    tasks = trace.sessions[0].tasks
    assert len(tasks) == 2
    ids_per_task = [{c.span_id for c in t.llm_calls} for t in tasks]
    assert {"msg_main"} in ids_per_task
    assert {"msg_side1", "msg_side2"} in ids_per_task


def test_resolve_adapter_detects_format_by_content_not_extension(tmp_path: Path) -> None:
    """Both dialects use .json/.jsonl, so detection must sniff content: a
    session JSONL resolves to the Claude Code adapter, while OTLP files —
    including OTLP JSON Lines, which share the .jsonl suffix — resolve to
    the OTLP adapter."""
    session_path = _write_session(tmp_path, [_assistant("msg_01", "u1")])
    otlp_path = tmp_path / "trace.json"
    otlp_path.write_text(json.dumps({"resourceSpans": []}), encoding="utf-8")
    otlp_lines_path = tmp_path / "trace-lines.jsonl"
    otlp_lines_path.write_text(json.dumps({"resourceSpans": []}) + "\n", encoding="utf-8")

    assert isinstance(resolve_adapter(str(session_path)), ClaudeCodeAdapter)
    assert isinstance(resolve_adapter(str(otlp_path)), OTLPFileAdapter)
    assert isinstance(resolve_adapter(str(otlp_lines_path)), OTLPFileAdapter)


def test_find_latest_session_picks_most_recently_modified_file(tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    older = projects / "proj1" / "a.jsonl"
    newer = projects / "proj2" / "b.jsonl"
    for path in (older, newer):
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    assert find_latest_session(projects_dir=projects) == newer


def test_find_latest_session_returns_none_for_missing_dir(tmp_path: Path) -> None:
    assert find_latest_session(projects_dir=tmp_path / "does-not-exist") is None
