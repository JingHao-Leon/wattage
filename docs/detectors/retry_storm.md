# retry_storm

**Detects:** the same request re-sent back-to-back — a client retry loop billing the full prompt every attempt.

A 429 loop, a timeout loop, or a schema-validation loop doesn't look like an error in the trace: it looks like a busy agent. But every attempt re-bills the entire prompt, and three attempts at a 100k-token context cost three full prompts for one answer.

## How it works

The signature is structural: a run of **consecutive same-model calls whose prompt-side token counts all stay within `shape_tolerance` (default 2%) of the run's first call, with no tool activity between attempts**. A tool call between two LLM calls means the agent acted on the response — a healthy iterating loop, not a retry — so it breaks the run; and anchoring the tolerance to the first call stops it chaining, so a long session whose context grows ~1% per turn never reads as one endless storm. `min_repeats` (default 3) or more matching attempts is not what a healthy agent step looks like. Runs of five or more are high severity.

The dollars reported are the **real billed cost of the repeat attempts** (everything after the first call in the run) — not a chars-per-token projection. But whether those attempts were *retries* rather than deliberate same-shaped calls is a heuristic judgment, so the finding's basis is `estimated`: shown in every report with a `~`, never able to move the grade or fail a CI build on its own.

## Known limitation

A deliberate fixed-prompt sampling loop (same prompt, N samples) has exactly this signature and would be flagged. If that's your agent's design, raise `retry_storm.min_repeats` or disable the detector — the finding text says so too. There is no reliable way to distinguish "retry" from "resample" from token counts alone, which is precisely why this detector's dollars stay out of the grade.

## Fix

Add backoff with a retry cap, and fix the underlying failure (schema validation, rate limit, timeout) instead of resending the full prompt. If the repeated context is unavoidable, cache the stable prefix so retries at least re-read it at the cached rate.

## Quality risk: none

Capping runaway retries doesn't change what the model sees on a successful attempt.
