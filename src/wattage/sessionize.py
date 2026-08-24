"""Raw spans -> Trace(Session(Task(Loop(Iteration)))), per doc §7.4/§5.6.

Sessions are grouped by trace_id. Within a session, *every* `invoke_agent`
span becomes its own Task boundary — including nested ones (an agent
invoking a sub-agent), so each gets its own convergence analysis rather than
being flattened into its parent's loop (Phase 2's "fully implemented" loop
reconstruction). Call spans not claimed by any agent span (either because
there are no agent spans at all, or because some calls sit outside every
agent's subtree) fall into a final catch-all task so nothing is silently
dropped.

Within a task, the loop-reconstruction fallback heuristic (§5.6) applies: a
chat span opens a new iteration, and any execute_tool/embeddings spans that
follow before the next chat span join that iteration. If the task's call
spans include any tool/retrieval activity, the whole sequence becomes a
single Loop of iterations; a task with pure chat (no tool/retrieval calls at
all) has no loop — those calls sit directly on Task.llm_calls.

Loop.reached_success is set by a structural heuristic (see
_infer_reached_success) — not a semantic guarantee, and not read by
nonconvergence's classification anymore (see detectors/convergence.py):
it's true for any loop ending in a plain chat response, which includes
both a genuine success and an agent simply giving up with an
uninformative final answer. Kept as informational metadata only.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from wattage.adapters.base import AdapterError
from wattage.models import Iteration, Loop, RawSpan, Session, SpanKind, Task, Trace
from wattage.normalize import normalize_llm_call, normalize_retrieval_call, normalize_tool_call

_CALL_KINDS = {SpanKind.chat, SpanKind.execute_tool, SpanKind.embeddings}

# Kinds whose normalized form carries no TokenUsage: any gen_ai.usage.*
# attributes on them are never priced, so they count toward
# dropped_token_spans (see sessionize()).
_UNPRICED_USAGE_KINDS = {
    SpanKind.other,
    SpanKind.execute_tool,
    SpanKind.embeddings,
    SpanKind.create_agent,
}

_USAGE_ATTR_PREFIX = "gen_ai.usage."

_SESSION_ID_KEYS = ("session.id", "gen_ai.conversation.id")


def _carries_usage(span: RawSpan) -> bool:
    return any(key.startswith(_USAGE_ATTR_PREFIX) for key in span.attributes)


def _session_key(span: RawSpan) -> str:
    # Production agents emit one OTel trace per user turn; the semconv
    # conversation id (or the common session.id convention) stitches a
    # multi-turn conversation back into one session, so cross-turn analysis
    # doesn't shatter. Fall back to trace_id, then to the span's own id.
    for key in _SESSION_ID_KEYS:
        value = span.attributes.get(key)
        if isinstance(value, str) and value:
            return value
    return span.trace_id or span.span_id


def sessionize(spans: Iterable[RawSpan], source: str) -> Trace:
    seen_ids: set[tuple[str, str]] = set()
    duplicate_spans = 0
    dropped_token_spans = 0
    all_spans: list[RawSpan] = []
    for span in spans:
        # OTLP delivery is at-least-once: a re-exported trace can contain the
        # same span twice, and counting it twice double-bills every one of
        # its dollars. First occurrence wins; the collapse is counted and
        # surfaced, never silent. Keyed on (trace_id, span_id): span ids are
        # only unique WITHIN a trace, and a multi-trace export whose emitter
        # uses counter-style ids ('1', '2', ...) must not have distinct
        # traces' genuinely different calls collapsed as "duplicates".
        key = (span.trace_id or "", span.span_id)
        if key in seen_ids:
            duplicate_spans += 1
            continue
        seen_ids.add(key)
        all_spans.append(span)
        # A span carrying gen_ai.usage.* token attributes whose usage will
        # never be priced would silently vanish from every total — the
        # exact undercount this project exists to prevent. That covers
        # unclassifiable spans (kind other) AND tool/embedding/create_agent
        # spans, whose normalized forms carry no usage fields. (invoke_agent
        # spans with usage are handled in _build_session, where it's known
        # whether their subtree's calls carried the real per-call numbers.)
        if span.kind in _UNPRICED_USAGE_KINDS and _carries_usage(span):
            dropped_token_spans += 1

    by_session: dict[str, list[RawSpan]] = defaultdict(list)
    for span in all_spans:
        by_session[_session_key(span)].append(span)

    # One malformed span (a negative token count, an impossible usage shape)
    # must not abort a ten-thousand-span CI run: the span is quarantined and
    # reported, and analysis proceeds on everything valid. If nothing valid
    # remains, the report layer refuses to grade — with the quarantine
    # reasons in the error, not a generic "empty trace".
    skipped: list[str] = []
    dropped_agent_usage = [0]
    sessions = [
        _build_session(session_id, session_spans, skipped, dropped_agent_usage)
        for session_id, session_spans in by_session.items()
    ]
    return Trace(
        source=source,
        sessions=sessions,
        dropped_token_spans=dropped_token_spans + dropped_agent_usage[0],
        duplicate_spans=duplicate_spans,
        skipped_spans=skipped,
    )


def _build_session(
    session_id: str,
    spans: list[RawSpan],
    skipped: list[str],
    dropped_agent_usage: list[int],
) -> Session:
    by_id = {s.span_id: s for s in spans}
    children: dict[str | None, list[str]] = defaultdict(list)
    for s in spans:
        children[s.parent_span_id].append(s.span_id)

    agent_ids = {s.span_id for s in spans if s.kind == SpanKind.invoke_agent}
    agent_spans = sorted((by_id[sid] for sid in agent_ids), key=lambda s: s.start_ns)

    tasks: list[Task] = []
    claimed: set[str] = set()
    for i, agent_span in enumerate(agent_spans):
        descendant_ids = _descendant_call_ids(agent_span.span_id, children, agent_ids)
        call_spans = sorted(
            (by_id[cid] for cid in descendant_ids if by_id[cid].kind in _CALL_KINDS),
            key=lambda s: s.start_ns,
        )
        claimed.update(cid for cid in descendant_ids)
        task = _build_task(f"{session_id}:task{i}", call_spans, skipped)
        # An agent span carrying aggregate gen_ai.usage.* whose subtree
        # produced NO priced calls is real spend with nowhere to go
        # (OpenInference AGENT roots report run totals this way) — counted
        # so the undercount is loud. When the subtree DOES have chat calls,
        # the aggregate is a duplicate of numbers already counted per-call,
        # and skipping it silently is correct.
        if _carries_usage(agent_span) and not any(s.kind == SpanKind.chat for s in call_spans):
            dropped_agent_usage[0] += 1
        tasks.append(task)

    leftover = sorted(
        (s for s in spans if s.kind in _CALL_KINDS and s.span_id not in claimed),
        key=lambda s: s.start_ns,
    )
    if leftover:
        tasks.append(_build_task(f"{session_id}:task{len(tasks)}", leftover, skipped))

    return Session(session_id=session_id, tasks=tasks)


def _descendant_call_ids(
    root_id: str, children: dict[str | None, list[str]], agent_ids: set[str]
) -> list[str]:
    """Descendants of root_id, not recursing past a *nested* agent span (its
    subtree belongs to its own task, not this one)."""
    result: list[str] = []
    stack = list(children.get(root_id, []))
    while stack:
        cid = stack.pop()
        result.append(cid)
        if cid not in agent_ids:
            stack.extend(children.get(cid, []))
    return result


def _quarantine(span: RawSpan, exc: AdapterError, skipped: list[str]) -> None:
    skipped.append(f"span {span.span_id}: {exc}")


def _build_task(task_id: str, call_spans: list[RawSpan], skipped: list[str]) -> Task:
    has_tool_or_retrieval = any(
        s.kind in (SpanKind.execute_tool, SpanKind.embeddings) for s in call_spans
    )
    if not has_tool_or_retrieval:
        llm_calls = []
        for s in call_spans:
            if s.kind != SpanKind.chat:
                continue
            try:
                llm_calls.append(normalize_llm_call(s))
            except AdapterError as exc:
                _quarantine(s, exc, skipped)
        return Task(task_id=task_id, llm_calls=llm_calls)

    groups: list[list[RawSpan]] = []
    for s in call_spans:
        if s.kind == SpanKind.chat or not groups:
            groups.append([s])
        else:
            groups[-1].append(s)

    iterations = []
    for idx, group in enumerate(groups):
        iteration = Iteration(index=idx)
        for s in group:
            try:
                if s.kind == SpanKind.chat:
                    iteration.llm_calls.append(normalize_llm_call(s))
                elif s.kind == SpanKind.execute_tool:
                    iteration.tool_calls.append(normalize_tool_call(s))
                elif s.kind == SpanKind.embeddings:
                    iteration.retrievals.append(normalize_retrieval_call(s))
            except AdapterError as exc:
                _quarantine(s, exc, skipped)
        iterations.append(iteration)

    loop = Loop(
        loop_id=f"{task_id}:loop0",
        iterations=iterations,
        reached_success=_infer_reached_success(iterations),
    )
    return Task(task_id=task_id, loops=[loop])


def _infer_reached_success(iterations: list[Iteration]) -> bool:
    """Structural proxy, not a semantic guarantee: a loop "reached success"
    if its last iteration produced LLM output with no further tool call
    pending — i.e. the agent stopped calling tools rather than being cut off
    mid-action. Genuine goal-completion detection needs semantics (an
    explicit goal signal or the optional judge), which this doesn't have —
    this is equally true whether the agent actually solved the task or
    just gave up with a final "I couldn't do this" answer, which is why
    nonconvergence no longer treats this flag as a classification gate
    (see detectors/convergence.py). Retained as informational metadata.
    """
    if not iterations:
        return False
    last = iterations[-1]
    return bool(last.llm_calls) and not last.tool_calls
