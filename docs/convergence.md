# The Convergence Engine

This is Wattage's technical moat: the `nonconvergence` detector catches agents that are stuck in a loop, spinning through iterations without actually making progress toward their goal — and it catches patterns that the "shallow loop guards" already built into several agent frameworks structurally cannot see.

## Why shallow loop guards aren't enough

The most common defense against runaway agent loops today is exact-match duplicate detection: hash each tool call's name and arguments, keep a sliding window of recent hashes, and flag it if you see the same hash twice. This is cheap and it catches the most obvious case — but it has three specific blind spots:

- **Fuzzy loops.** An agent retrying the same failing action with a slightly different argument each time (an incrementing attempt counter, a fresh timestamp) never hashes equal to its own prior attempt, even though it's the same unproductive retry in every way that matters.
- **Semantic oscillation.** An agent alternating between two strategies — `check_status` → `retry_action` → `check_status` → `retry_action` — with each individual call looking "new" relative to the *immediately* preceding one.
- **Productive-looking stalls.** An agent whose every tool call is technically unique (different tool, different arguments, every single time) but whose actual results are boilerplate and whose context keeps growing regardless. An exact-match detector sees zero duplicates and concludes everything is fine.

## What Wattage measures instead

Rather than asking "have I seen this exact call before," Wattage measures **progress** directly, per iteration, using five signals:

- **Evidence gain (E)** — how novel this iteration's tool/retrieval results are against everything seen so far in the loop (cosine novelty over a lightweight embedding).
- **State delta (S)** — how much the agent's chosen action actually changed from just the *immediately prior* iteration (a local, not cumulative, comparison — the thing that keeps this a distinct signal from E).
- **Oscillation (O)** — periodic repetition in the sequence of actions taken (period 2 or more only; a single repeated action is `thrashing`'s signature, not oscillation's), robust to argument noise because it keys on tool names, not exact arguments.
- **Growth-vs-information (G)** — how many new context tokens this iteration cost relative to how much genuinely new evidence arrived, independent of E, so a loop can have low evidence gain yet still be assessed on whether that repetition was *cheap* or *expensive*.
- **Goal proximity (P)** — only meaningful when an explicit goal signal is supplied; defaults to a neutral value otherwise, never a guess.

These combine into a per-iteration progress score, and a loop is classified as **productive**, **thrashing** (the same unproductive action repeated), **oscillating** (a detected cycle), or **stalled** (context keeps growing, evidence and state both stay flat — the exact-match blind spot). A loop that genuinely recovered — real, substantive, different content in its later iterations — is never flagged for how it got there; but that has to be earned by actual observable content, not merely by the loop ending. A chat-only final answer with no captured message text (the normal case — message content isn't captured by the real adapter) is never enough on its own to exempt a loop: a loop that thrashed the whole time and then simply gave up with an uninformative answer is still flagged, exactly as if it had never answered at all.

## The benchmark — and exactly what it shows

The project built a hand-reviewed set of 10 labeled synthetic loops (`benchmarks/adversarial_fixtures.py`) — three productive, three thrashing, two oscillating, two stalled — and ran both Wattage's classifier and a real reference implementation of the SHA-256 exact-match baseline against it (`benchmarks/harness.py`).

**Wattage: F1 1.00. The SHA-256 baseline: F1 0.25** (recall 0.14 — it catches exactly one of the seven genuinely non-convergent cases: the one where the arguments happened to be byte-identical every time). The concrete example the baseline misses entirely: a loop where every single tool call is technically unique, yet Wattage correctly identifies it as stalled because the context kept growing while the results stayed boilerplate.

Read those numbers for what they are, because the details matter:

- **The suite is small (n=10) and written by this project.** Fixture labels were verified against the classifier during development, so a perfect score on it is closer to a *regression suite passing* than an independent field study. It is presented here as evidence about *blind spots*, not as a general accuracy claim.
- **The positive cases were deliberately constructed to defeat exact-match hashing** (incrementing attempt counters, rephrased retries, cycling strategies). That's the point — these are the real-world patterns a hash can structurally never catch — but it means the baseline's 0.14 recall is by construction, demonstrating the blind-spot class rather than measuring average-case performance.
- **The published numbers come from the dependency-free hash-embedder tier** (the default install, and what CI runs). The optional `sentence-transformers` tier produces different similarity distributions; the classifier's thresholds were tuned on the hash tier.

What would upgrade this from a blind-spot demonstration to a field benchmark — externally-labeled real agent loops, stronger baselines (fuzzy canonicalization, tool-name repetition counting), per-tier results — is exactly the kind of contribution the project would love help with.

Run it yourself; the table reproduces exactly, no hidden setup:

```bash
uv run python -m benchmarks.harness
```

## Embeddings: local by default, no API key required

The novelty computation needs *some* notion of text similarity. Rather than requiring a heavy ML dependency just to get a useful default, Wattage ships a dependency-free character n-gram hashing embedder that works out of the box, and upgrades automatically to `sentence-transformers` (materially better novelty detection) if you `pip install wattage[embeddings]` — the convergence engine works either way; only the quality of the "which loops are borderline" judgment changes.

## Severity, and the critical escalation

A non-convergent loop's finding carries the real billed dollars of every iteration after its last productive one — measured, not estimated, so it counts toward the grade and the CI gate. When that post-productive waste reaches half the loop's own total spend (`nonconvergence.critical_ratio`, default 0.5), the finding escalates to **critical** severity — which is what trips `fail_on: any_critical` and fails the build outright. A runaway loop that burned half its budget going nowhere is exactly the case a hard stop exists for.

## What's next

A sampled LLM judge for genuinely ambiguous cases exists in the codebase (`convergence/judge.py`, tested) but is **not wired into the analysis pipeline** — configuring `judge: sampled` is a hard config error rather than a silent no-op, and the docs won't pretend otherwise until it's actually invokable. Runtime enforcement — the same signals running live to warn, then nudge, then hard-stop a thrashing agent in-flight — is the natural next step once this diagnostic layer has proven itself on real traces, which is exactly the sequencing this project is following.
