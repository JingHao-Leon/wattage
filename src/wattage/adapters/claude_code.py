"""Claude Code session adapter: ~/.claude/projects session JSONL -> RawSpans.

Claude Code writes every session as a JSON Lines file under
``~/.claude/projects/<project-dir>/<session-id>.jsonl``. Each line is one
record; the ones that matter here:

- ``type: "assistant"`` records carry one API response's ``message`` with
  ``model``, ``usage`` (input/output/cache_read/cache_creation token counts,
  ``output_tokens_details.thinking_tokens`` for reasoning, and a
  ``cache_creation`` 5m/1h TTL breakdown the OTel GenAI attributes can't
  even express), and ``content`` blocks including ``tool_use``. One API
  response is often split across SEVERAL consecutive records — one per
  content block, each repeating the same ``message.id`` and the same usage
  object — so records are aggregated by message id first; counting each
  record would overbill a response by up to the number of its blocks.
- ``type: "user"`` records carry ``tool_result`` blocks (matched back to
  their ``tool_use`` by id) — the tool outputs that feed the convergence
  and redundancy detectors.
- Everything else (mode, permission-mode, attachment, file-history-snapshot,
  summary, ...) is session bookkeeping with no cost dimension: skipped.

Sub-agent activity (``isSidechain: true``) is grouped into its own task per
sidechain root, so a sub-agent's loop is analyzed on its own rather than
interleaved into the main conversation's iteration sequence.

Honesty note: token costs computed from these sessions are standard
Anthropic API rates. For subscription (Pro/Max) users that is the
API-equivalent value of the session, not an invoice line — the report says
so rather than implying otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from wattage.adapters.base import Adapter, AdapterError
from wattage.models import RawSpan, SpanKind

# Claude Code emits placeholder "<synthetic>" model entries for locally
# generated messages (no API call, no cost) — never priced, never counted.
_SYNTHETIC_MODEL = "<synthetic>"

_DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def find_latest_session(projects_dir: Path | None = None) -> Path | None:
    """The most recently modified session file across all Claude Code
    projects — what `wattage report --claude-code` analyzes by default."""
    root = projects_dir or _DEFAULT_PROJECTS_DIR
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def looks_like_session_line(line: str) -> bool:
    """Cheap content sniff: does this JSONL line look like a Claude Code
    session record (vs. an OTLP JSON Lines trace)?"""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(record, dict):
        return False
    return "type" in record and ("sessionId" in record or "session_id" in record)


# Session files for compacted/linked sessions START with summary records
# ({"type":"summary",...}) that carry no sessionId — sniffing only line 1
# would misroute those genuine session files to the OTLP adapter. A handful
# of lines is enough to hit a real user/assistant record.
_SNIFF_LINES = 25


def _looks_like_session_file(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            checked = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if looks_like_session_line(line):
                    return True
                checked += 1
                if checked >= _SNIFF_LINES:
                    return False
    except OSError:
        return False
    return False


def _timestamp_ns(value: Any) -> int:
    if not isinstance(value, str) or not value:
        return 0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1_000_000_000)


def _result_text(block: dict[str, Any]) -> str | None:
    """tool_result content is either a plain string or a list of content
    blocks; extract the text either way."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        joined = "\n".join(part for part in parts if part)
        return joined or None
    return None


class _ApiCall:
    """One aggregated API response (all JSONL records sharing a message id)."""

    def __init__(self, record: dict[str, Any], message: dict[str, Any]) -> None:
        self.message_id: str = message.get("id") or record.get("uuid") or ""
        self.model: str = str(message.get("model") or "unknown")
        self.usage: dict[str, Any] = message.get("usage") or {}
        self.first_ts_ns: int = _timestamp_ns(record.get("timestamp"))
        self.last_ts_ns: int = self.first_ts_ns
        self.sidechain: bool = bool(record.get("isSidechain"))
        self.uuid: str = str(record.get("uuid") or "")
        self.parent_uuid: str | None = record.get("parentUuid")
        self.tool_uses: list[dict[str, Any]] = []
        self._absorb_content(message)

    def absorb(self, record: dict[str, Any], message: dict[str, Any]) -> None:
        ts = _timestamp_ns(record.get("timestamp"))
        if ts:
            self.last_ts_ns = max(self.last_ts_ns, ts)
            if not self.first_ts_ns:
                self.first_ts_ns = ts
        # Later records for the same response carry the same (final) usage;
        # take the latest seen so a streaming update wins.
        usage = message.get("usage")
        if usage:
            self.usage = usage
        self._absorb_content(message)

    def _absorb_content(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if not any(t.get("id") == block.get("id") for t in self.tool_uses):
                    self.tool_uses.append(block)


class ClaudeCodeAdapter(Adapter):
    format_name = "claude-code"

    def supports(self, source: str) -> bool:
        path = Path(source)
        if path.is_dir():
            return any(
                _looks_like_session_file(candidate)
                for candidate in sorted(path.glob("*.jsonl"))[:5]
            )
        if not source.endswith(".jsonl") or not path.is_file():
            return False
        return _looks_like_session_file(path)

    def read(self, source: str | IO[str]) -> Iterable[RawSpan]:
        if not isinstance(source, str):
            yield from self._read_lines(source, session_hint="stream")
            return
        path = Path(source)
        if path.is_dir():
            files = sorted(path.glob("*.jsonl"))
            if not files:
                raise AdapterError(f"'{source}' contains no .jsonl session files")
            # Real project directories routinely contain sessions with no
            # priced API calls (quit before the first reply, or only
            # "<synthetic>" error placeholders) — one such file must not
            # abort the whole project analysis. Each file's spans are
            # buffered so a mid-file error can't leave a half-emitted
            # session; a file-level error skips just that file, and the
            # read only fails if NO file yielded anything.
            errors: list[str] = []
            yielded_any = False
            for file in files:
                try:
                    with open(file, encoding="utf-8") as f:
                        spans = list(self._read_lines(f, session_hint=file.stem))
                except AdapterError as exc:
                    errors.append(f"{file.name}: {exc}")
                    continue
                yielded_any = True
                yield from spans
            if not yielded_any:
                detail = "; ".join(errors[:3]) or "no session files could be read"
                raise AdapterError(f"'{source}' has no readable sessions ({detail})")
            return
        with open(path, encoding="utf-8") as f:
            yield from self._read_lines(f, session_hint=path.stem)

    def _read_lines(self, lines: IO[str], session_hint: str) -> Iterator[RawSpan]:
        calls: dict[str, _ApiCall] = {}
        call_order: list[str] = []
        tool_results: dict[str, tuple[str | None, int]] = {}  # tool_use_id -> (text, ts_ns)
        record_by_uuid: dict[str, dict[str, Any]] = {}
        session_id = session_hint

        for line_no, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AdapterError(
                    f"malformed session file: line {line_no} is not valid JSON ({exc})"
                ) from exc
            if not isinstance(record, dict):
                continue
            if isinstance(record.get("uuid"), str):
                record_by_uuid[record["uuid"]] = record
            sid = record.get("sessionId") or record.get("session_id")
            if isinstance(sid, str) and sid:
                session_id = sid

            record_type = record.get("type")
            if record_type == "assistant":
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                if message.get("model") == _SYNTHETIC_MODEL:
                    continue
                message_id = message.get("id") or record.get("uuid")
                if not isinstance(message_id, str) or not message_id:
                    continue
                existing = calls.get(message_id)
                if existing is None:
                    calls[message_id] = _ApiCall(record, message)
                    call_order.append(message_id)
                else:
                    existing.absorb(record, message)
            elif record_type == "user":
                message = record.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                ts = _timestamp_ns(record.get("timestamp"))
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_use_id = block.get("tool_use_id")
                            if isinstance(tool_use_id, str):
                                tool_results[tool_use_id] = (_result_text(block), ts)

        if not call_order:
            raise AdapterError(
                f"'{session_hint}' contains no priced API calls "
                "(no assistant records with usage data)"
            )

        yield from self._to_spans(calls, call_order, tool_results, record_by_uuid, session_id)

    def _to_spans(
        self,
        calls: dict[str, _ApiCall],
        call_order: list[str],
        tool_results: dict[str, tuple[str | None, int]],
        record_by_uuid: dict[str, dict[str, Any]],
        session_id: str,
    ) -> Iterator[RawSpan]:
        # Synthetic agent roots: one for the main conversation, one per
        # sidechain root — so sub-agent loops become their own tasks.
        main_agent_id = f"cc-agent:{session_id}:main"
        emitted_agents: set[str] = set()

        def agent_span(agent_id: str, name: str, start_ns: int) -> RawSpan:
            return RawSpan(
                span_id=agent_id,
                parent_span_id=None,
                trace_id=session_id,
                name=name,
                kind=SpanKind.invoke_agent,
                attributes={"gen_ai.operation.name": "invoke_agent"},
                start_ns=start_ns,
                end_ns=start_ns,
            )

        def sidechain_root(call: _ApiCall) -> str:
            """Walk the parentUuid chain to this sidechain's first record."""
            seen: set[str] = set()
            current = call.uuid
            root = current
            while current and current not in seen:
                seen.add(current)
                record = record_by_uuid.get(current)
                if record is None or not record.get("isSidechain"):
                    break
                root = current
                parent = record.get("parentUuid")
                current = parent if isinstance(parent, str) else ""
            return root

        for message_id in call_order:
            call = calls[message_id]
            usage = call.usage
            if call.sidechain:
                agent_id = f"cc-agent:{session_id}:side:{sidechain_root(call)}"
                agent_name = "invoke_agent claude-code-subagent"
            else:
                agent_id = main_agent_id
                agent_name = "invoke_agent claude-code"
            if agent_id not in emitted_agents:
                emitted_agents.add(agent_id)
                yield agent_span(agent_id, agent_name, call.first_ts_ns)

            # Usage values pass through RAW: normalize._as_token_count is the
            # single validation point for token counts, and it quarantines a
            # malformed value (a string, a float, a negative) with a visible
            # skipped-span report. Coercing bad values to 0 here would turn
            # corruption into a confident $0 line item — the exact
            # plausible-looking-wrong-number this project exists to avoid.
            attrs: dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": call.model,
                "gen_ai.usage.input_tokens": usage.get("input_tokens"),
                "gen_ai.usage.output_tokens": usage.get("output_tokens"),
                "gen_ai.usage.cache_read.input_tokens": usage.get("cache_read_input_tokens"),
                "gen_ai.usage.cache_creation.input_tokens": usage.get(
                    "cache_creation_input_tokens"
                ),
                "session.id": session_id,
            }
            details = usage.get("output_tokens_details")
            if isinstance(details, dict) and details.get("thinking_tokens") is not None:
                attrs["gen_ai.usage.reasoning.output_tokens"] = details["thinking_tokens"]
            ttl = usage.get("cache_creation")
            if isinstance(ttl, dict):
                attrs["gen_ai.usage.cache_creation.ephemeral_5m_input_tokens"] = ttl.get(
                    "ephemeral_5m_input_tokens"
                )
                attrs["gen_ai.usage.cache_creation.ephemeral_1h_input_tokens"] = ttl.get(
                    "ephemeral_1h_input_tokens"
                )

            yield RawSpan(
                span_id=message_id,
                parent_span_id=agent_id,
                trace_id=session_id,
                name=f"chat {call.model}",
                kind=SpanKind.chat,
                attributes=attrs,
                start_ns=call.first_ts_ns,
                end_ns=call.last_ts_ns,
            )

            # Tool executions requested by this response, ordered just after
            # the chat span so sessionize's iteration heuristic binds them
            # to it.
            for offset, block in enumerate(call.tool_uses, start=1):
                tool_use_id = block.get("id")
                if not isinstance(tool_use_id, str) or not tool_use_id:
                    continue
                result_text, result_ts = tool_results.get(tool_use_id, (None, 0))
                tool_attrs: dict[str, Any] = {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": str(block.get("name") or "unknown"),
                    "session.id": session_id,
                }
                tool_input = block.get("input")
                if tool_input is not None:
                    tool_attrs["gen_ai.tool.call.arguments"] = json.dumps(
                        tool_input, sort_keys=True, default=str
                    )
                if result_text is not None:
                    tool_attrs["gen_ai.tool.call.result"] = result_text
                start_ns = call.last_ts_ns + offset
                yield RawSpan(
                    span_id=tool_use_id,
                    parent_span_id=agent_id,
                    trace_id=session_id,
                    name=f"execute_tool {tool_attrs['gen_ai.tool.name']}",
                    kind=SpanKind.execute_tool,
                    attributes=tool_attrs,
                    start_ns=start_ns,
                    end_ns=result_ts or start_ns,
                )
