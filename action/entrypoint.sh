#!/usr/bin/env bash
# Entrypoint for the root action.yml. Runs `wattage ci`, emits report fields
# as step outputs, writes the report to the step summary, posts/updates a
# sticky PR comment when running on a pull_request event, and always
# surfaces the real exit code (0-4, per doc §11.3) as both the action's own
# exit status and the `exit-code` output — never swallowed, never guessed at.
set -uo pipefail

ARGS=("$INPUT_SOURCE")
[ -n "${INPUT_BASELINE:-}" ] && ARGS+=(--baseline "$INPUT_BASELINE")
[ -n "${INPUT_PRICING:-}" ] && ARGS+=(--pricing "$INPUT_PRICING")
[ -n "${INPUT_QUALITY:-}" ] && ARGS+=(--quality "$INPUT_QUALITY")
[ -n "${INPUT_FAIL_ON:-}" ] && ARGS+=(--fail-on "$INPUT_FAIL_ON")
[ -n "${INPUT_CONFIG:-}" ] && ARGS+=(--config "$INPUT_CONFIG")

PR_COMMENT_FILE="$(mktemp)"
ARGS+=(--pr-comment-out "$PR_COMMENT_FILE")

SARIF_OUT="${INPUT_SARIF_OUT:-wattage.sarif}"
ARGS+=(--sarif-out "$SARIF_OUT")

[ -n "${INPUT_JUNIT_OUT:-}" ] && ARGS+=(--junit-out "$INPUT_JUNIT_OUT")
[ -n "${INPUT_BADGE_OUT:-}" ] && ARGS+=(--badge-out "$INPUT_BADGE_OUT")

REPORT_JSON="${RUNNER_TEMP:-$(mktemp -d)}/wattage-report.json"
# A previous action invocation in this same job may have left a report at
# this fixed path; a later run that fails before producing one must emit
# EMPTY outputs, not the previous trace's numbers.
rm -f "$REPORT_JSON"
ARGS+=(--report-json-out "$REPORT_JSON")

wattage ci "${ARGS[@]}"
EXIT_CODE=$?

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "exit-code=$EXIT_CODE" >> "$GITHUB_OUTPUT"
  if [ -s "$REPORT_JSON" ]; then
    {
      echo "score=$(jq -r '.score.efficiency // empty' "$REPORT_JSON")"
      echo "grade=$(jq -r '.score.grade // empty' "$REPORT_JSON")"
      echo "total-dollars=$(jq -r '.total_dollars // empty' "$REPORT_JSON")"
      echo "recoverable-dollars=$(jq -r '.score.recoverable_dollars // empty' "$REPORT_JSON")"
    } >> "$GITHUB_OUTPUT"
  else
    # Early failures (exit 2 config / exit 3 ingestion) produce no report;
    # emit empty outputs so downstream `steps.*.outputs.*` reads are defined.
    printf '%s\n' "score=" "grade=" "total-dollars=" "recoverable-dollars=" \
      >> "$GITHUB_OUTPUT"
  fi
fi

# Always publish the report to the step summary — this works on every run,
# including fork PRs and permission-restricted repos, with zero
# configuration, so the report always lands somewhere visible.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ] && [ -s "$PR_COMMENT_FILE" ]; then
  cat "$PR_COMMENT_FILE" >> "$GITHUB_STEP_SUMMARY"
fi

# Sticky PR comment: a hidden marker identifies wattage's own comment so
# each push updates it in place instead of spamming a new comment.
MARKER="<!-- wattage-report -->"
should_comment="${INPUT_PR_COMMENT:-true}"
if [ "$should_comment" = "true" ] \
  && [ "${GITHUB_EVENT_NAME:-}" = "pull_request" ] \
  && [ -s "$PR_COMMENT_FILE" ]; then
  PR_NUMBER=$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH")
  BODY_FILE="$(mktemp)"
  { echo "$MARKER"; cat "$PR_COMMENT_FILE"; } > "$BODY_FILE"

  COMMENT_ID=$(gh api --paginate "repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/comments" \
    --jq ".[] | select(.body | contains(\"$MARKER\")) | .id" 2>/dev/null \
    | head -n 1)

  comment_posted=true
  if [ -n "$COMMENT_ID" ]; then
    gh api --method PATCH \
      "repos/$GITHUB_REPOSITORY/issues/comments/$COMMENT_ID" \
      -F body=@- < "$BODY_FILE" > /dev/null || comment_posted=false
  else
    gh pr comment "$PR_NUMBER" --body-file "$BODY_FILE" \
      --repo "$GITHUB_REPOSITORY" || comment_posted=false
  fi
  if [ "$comment_posted" != "true" ]; then
    # Not fatal (the gate's verdict is the exit code, not the comment), but
    # never silent either: say what happened and where the report still is.
    echo "::warning::wattage could not post the PR comment (this usually" \
      "means the workflow's GITHUB_TOKEN lacks 'pull-requests: write'," \
      "e.g. on a fork PR). The full report is in the job's step summary."
  fi
fi

exit "$EXIT_CODE"
