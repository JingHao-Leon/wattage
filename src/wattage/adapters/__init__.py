"""Adapter selection: pick the right reader for a trace source by looking at
its content, not its file extension — two .json/.jsonl dialects can't be told
apart by suffix.

Order matters and is deliberate: the Claude Code sniff is precise (a JSONL
line with `type` + `sessionId`), so it runs first; the OTLP adapter is the
general fallback for everything JSON-shaped, and its own error messages
handle the genuinely-unreadable case.
"""

from __future__ import annotations

from pathlib import Path

from wattage.adapters.base import Adapter, AdapterError
from wattage.adapters.claude_code import ClaudeCodeAdapter
from wattage.adapters.otlp_file import OTLPFileAdapter

__all__ = ["Adapter", "AdapterError", "ClaudeCodeAdapter", "OTLPFileAdapter", "resolve_adapter"]


def resolve_adapter(source: str) -> Adapter:
    path = Path(source)
    claude_code = ClaudeCodeAdapter()
    if path.is_dir():
        if claude_code.supports(source):
            return claude_code
        raise AdapterError(
            f"'{source}' is a directory but doesn't contain Claude Code session files — "
            "point wattage at a trace file, or at a ~/.claude/projects/<project> directory"
        )
    if claude_code.supports(source):
        return claude_code
    return OTLPFileAdapter()
