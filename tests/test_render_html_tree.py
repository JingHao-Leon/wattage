from wattage.models import Iteration, LLMCall, Loop, Session, Task, TokenUsage, ToolCall, Trace
from wattage.pricing.engine import PricingEngine
from wattage.pricing.registry import PricingRegistry
from wattage.render.html.tree import build_tree


def _llm(span_id: str, engine: PricingEngine, **usage_kwargs: int) -> LLMCall:
    call = LLMCall(
        span_id=span_id,
        provider="anthropic",
        model="claude-sonnet-4-6",
        usage=TokenUsage(**usage_kwargs),
        start_ns=0,
    )
    call.cost = engine.price_call(call)
    return call


def test_tree_root_matches_trace_source() -> None:
    trace = Trace(source="my-trace.json", sessions=[])
    tree = build_tree(trace)
    assert tree["name"] == "my-trace.json"
    assert tree["kind"] == "trace"
    assert tree["children"] == []


def test_single_child_structural_chains_collapse_into_one_row() -> None:
    """A single-session single-task trace used to render four visually
    identical full-width gray bars (trace -> session -> task -> loop) before
    anything informative appeared. Those chains now merge into one row whose
    name shows the path, so the shareable artifact leads with content."""
    engine = PricingEngine(PricingRegistry.load())
    call = _llm("l0", engine, input=1000, output=50)
    task = Task(task_id="t0", llm_calls=[call])
    trace = Trace(source="s", sessions=[Session(session_id="s0", tasks=[task])])

    tree = build_tree(trace)
    # trace -> session -> task all had exactly one structural child, so the
    # root IS the merged task row, and its children are the actual calls.
    assert tree["kind"] == "task"
    assert tree["name"] == "s › s0 › t0"
    assert tree["children"][0]["kind"] == "llm"


def test_multi_session_traces_do_not_collapse_across_sessions() -> None:
    engine = PricingEngine(PricingRegistry.load())

    def session(n: int) -> Session:
        task = Task(task_id=f"t{n}", llm_calls=[_llm(f"l{n}", engine, input=100)])
        return Session(session_id=f"s{n}", tasks=[task])

    trace = Trace(source="s", sessions=[session(1), session(2)])
    tree = build_tree(trace)
    assert tree["kind"] == "trace"
    assert len(tree["children"]) == 2
    # Each session -> task chain still collapses within itself.
    assert all(child["kind"] == "task" for child in tree["children"])


def test_llm_call_splits_into_only_nonzero_token_class_segments() -> None:
    engine = PricingEngine(PricingRegistry.load())
    call = _llm("l0", engine, input=1000, output=50, cache_read=200)
    task = Task(task_id="t0", llm_calls=[call])
    trace = Trace(source="s", sessions=[Session(session_id="s0", tasks=[task])])

    tree = build_tree(trace)
    llm_node = tree["children"][0]  # collapsed chain root -> llm
    assert llm_node["kind"] == "llm"
    categories = {c["category"] for c in llm_node["children"]}
    assert categories == {"input", "output", "cache_read"}
    # cache_creation and reasoning were zero, so they're absent, not zero-width entries.
    assert "cache_creation" not in categories
    assert "reasoning" not in categories


def test_llm_segment_tokens_and_dollars_match_the_call() -> None:
    engine = PricingEngine(PricingRegistry.load())
    call = _llm("l0", engine, input=1000, output=50)
    task = Task(task_id="t0", llm_calls=[call])
    trace = Trace(source="s", sessions=[Session(session_id="s0", tasks=[task])])

    tree = build_tree(trace)
    llm_node = tree["children"][0]
    input_seg = next(c for c in llm_node["children"] if c["category"] == "input")
    assert input_seg["tokens"] == 1000
    assert input_seg["dollars"] == call.cost.input
    # The segment carries its call's span id so a finding can light it up.
    assert input_seg["span_id"] == "l0"


def test_tool_call_estimates_tokens_from_result_length_and_has_no_dollars() -> None:
    tool_call = ToolCall(
        span_id="t0", name="search_docs", args={}, args_hash="x", result="x" * 40, start_ns=0
    )
    iteration = Iteration(index=0, tool_calls=[tool_call])
    loop = Loop(loop_id="loop0", iterations=[iteration], reached_success=False)
    task = Task(task_id="t0", loops=[loop])
    trace = Trace(source="s", sessions=[Session(session_id="s0", tasks=[task])])

    tree = build_tree(trace)
    # trace -> session -> task -> loop collapse into one row; iteration is
    # not part of the structural-collapse set, so it survives as a level.
    iteration_node = tree["children"][0]
    assert iteration_node["kind"] == "iteration"
    tool_node = iteration_node["children"][0]
    assert tool_node["kind"] == "tool"
    assert tool_node["dollars"] is None
    assert tool_node["tokens"] == 10  # 40 chars // 4
    assert tool_node["children"][0]["category"] == "tool_io"
    assert tool_node["span_id"] == "t0"


def test_tokens_roll_up_through_every_level() -> None:
    engine = PricingEngine(PricingRegistry.load())
    call = _llm("l0", engine, input=1000, output=50)
    task = Task(task_id="t0", llm_calls=[call])
    session = Session(session_id="s0", tasks=[task])
    trace = Trace(source="s", sessions=[session])

    tree = build_tree(trace)
    assert tree["tokens"] == 1050  # collapsed structural root
    assert tree["children"][0]["tokens"] == 1050  # the llm call
    segment_total = sum(seg["tokens"] for seg in tree["children"][0]["children"])
    assert segment_total == 1050
