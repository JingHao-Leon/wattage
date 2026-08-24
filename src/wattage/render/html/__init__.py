"""Self-contained HTML flame graph ("burn map", doc §3.2/§7.7): the
shareable, screenshot-worthy report. Everything — CSS, JS, data — is inlined
into one file; no CDN, no external font, no network call of any kind (doc
§7.7 requires this and §3.6 verifies it by scanning the output).

Sizing and category semantics are documented in tree.py; this module owns
the page shell (hero stat strip + token-breakdown bar), the flame-graph
rendering/interaction JS (pixel-space layout with a ResizeObserver — no
viewBox letterboxing), and the findings sidebar, which is wired to the
graph: clicking a finding lights up the exact frames whose spans burned
that money.
"""

from __future__ import annotations

import json
import re
from xml.sax.saxutils import escape

from wattage.models import FindingBasis, Report, Trace
from wattage.render.format import format_dollars
from wattage.render.html.tree import build_tree

_SEVERITY_STATUS = {
    "info": "warning",
    "low": "warning",
    "medium": "serious",
    "high": "critical",
    "critical": "critical",
}

_GRADE_CLASS = {"A": "grade-good", "B": "grade-good", "C": "grade-warn", "D": "grade-bad"}

# A lightning bolt as an inline SVG favicon — keeps the page fully
# self-contained (no external asset) while giving shared links an identity.
_FAVICON = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
    "%3Ctext y='.9em' font-size='90'%3E%E2%9A%A1%3C/text%3E%3C/svg%3E"
)

_TEMPLATE = (
    r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href=" """.strip()
    + _FAVICON
    + r"""">
<style>
  :root {
    --surface-1: #fcfcfb; --page: #f9f9f7; --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --cat-input: #2a78d6; --cat-cache-read: #1baf7a; --cat-cache-creation: #4a3aa7;
    --cat-reasoning: #eda100; --cat-output: #008300; --cat-tool-io: #eb6834;
    --status-good: #0ca30c; --status-warning: #fab219; --status-serious: #ec835a; --status-critical: #d03b3b;
    --neutral-frame: #efeee9; --neutral-frame-2: #e4e3dc; --neutral-frame-ink: #3a3936;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      --surface-1: #1a1a19; --page: #0d0d0d; --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --cat-input: #3987e5; --cat-cache-read: #199e70; --cat-cache-creation: #9085e9;
      --cat-reasoning: #c98500; --cat-output: #008300; --cat-tool-io: #d95926;
      --neutral-frame: #26251f; --neutral-frame-2: #302f27; --neutral-frame-ink: #d8d6cd;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --surface-1: #1a1a19; --page: #0d0d0d; --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --cat-input: #3987e5; --cat-cache-read: #199e70; --cat-cache-creation: #9085e9;
    --cat-reasoning: #c98500; --cat-output: #008300; --cat-tool-io: #d95926;
    --neutral-frame: #26251f; --neutral-frame-2: #302f27; --neutral-frame-ink: #d8d6cd;
    color-scheme: dark;
  }
  :root[data-theme="light"] {
    --surface-1: #fcfcfb; --page: #f9f9f7; --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --cat-input: #2a78d6; --cat-cache-read: #1baf7a; --cat-cache-creation: #4a3aa7;
    --cat-reasoning: #eda100; --cat-output: #008300; --cat-tool-io: #eb6834;
    --neutral-frame: #efeee9; --neutral-frame-2: #e4e3dc; --neutral-frame-ink: #3a3936;
    color-scheme: light;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink-1);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .num { font-variant-numeric: tabular-nums; font-family: ui-monospace, "SF Mono", Consolas, monospace; }
  header {
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
    padding: 18px 24px 14px; border-bottom: 1px solid var(--border);
  }
  header h1 { font-size: 18px; margin: 0; }
  header .source { color: var(--ink-muted); font-size: 13px; word-break: break-all; }
  header .spacer { flex: 1; }
  .score-chip {
    display: inline-flex; align-items: baseline; gap: 6px; padding: 4px 12px;
    border-radius: 999px; background: var(--neutral-frame); font-size: 13px;
  }
  .score-chip b { font-size: 15px; }
  .score-chip.grade-good b { color: var(--status-good); }
  .score-chip.grade-warn b { color: #9a6a00; }
  .score-chip.grade-bad b { color: var(--status-critical); }
  .score-chip.unpriced { background: var(--status-warning); color: #3a2a00; }
  .unpriced-banner {
    padding: 10px 24px; font-size: 13px; background: var(--status-warning); color: #3a2a00;
    border-bottom: 1px solid var(--border);
  }
  button.theme-toggle, button.reset-zoom {
    border: 1px solid var(--border); background: var(--surface-1); color: var(--ink-1);
    border-radius: 6px; padding: 5px 10px; font-size: 12px; cursor: pointer;
  }
  .stat-strip {
    display: flex; flex-wrap: wrap; gap: 28px; padding: 16px 24px 6px;
  }
  .stat { min-width: 110px; }
  .stat .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-muted); }
  .stat .stat-value { font-size: 22px; font-weight: 650; margin-top: 2px; }
  .stat .stat-sub { font-size: 11.5px; color: var(--ink-2); }
  .stat .stat-value.money-bad { color: var(--status-critical); }
  .stat .stat-value.money-good { color: var(--status-good); }
  .breakdown-wrap { padding: 8px 24px 16px; border-bottom: 1px solid var(--border); }
  .breakdown-bar {
    display: flex; height: 14px; border-radius: 7px; overflow: hidden;
    background: var(--neutral-frame);
  }
  .breakdown-bar span { display: block; height: 100%; }
  .breakdown-caption { font-size: 11.5px; color: var(--ink-2); margin-top: 6px; }
  main { display: flex; gap: 0; align-items: flex-start; flex-wrap: wrap; }
  .graph-panel { flex: 1 1 560px; min-width: 0; padding: 20px 24px; }
  .graph-toolbar { display: flex; align-items: center; gap: 10px; padding: 0 2px 10px; }
  .breadcrumb { font-size: 12px; color: var(--ink-2); flex: 1; min-width: 0; }
  .breadcrumb span { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; }
  .breadcrumb .sep { margin: 0 4px; color: var(--ink-muted); }
  .graph-box { border: 1px solid var(--border); border-radius: 8px; background: var(--surface-1); overflow: hidden; }
  svg#flamegraph { display: block; width: 100%; }
  .frame { stroke: var(--page); stroke-width: 1; cursor: pointer; transition: opacity .12s; }
  .frame.leaf { cursor: default; }
  .frame:hover { stroke: var(--ink-1); stroke-width: 1.5; }
  .frame.dimmed { opacity: 0.18; }
  .frame.hot { stroke: var(--status-critical); stroke-width: 2; }
  .frame-label {
    font-size: 11px; fill: var(--neutral-frame-ink); pointer-events: none;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
  }
  .frame-label.on-color { fill: #ffffff; }
  .frame-label.dimmed { opacity: 0.18; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; padding: 10px 2px 0; font-size: 12px; color: var(--ink-2); }
  .legend .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }
  #tooltip {
    position: fixed; pointer-events: none; z-index: 10; display: none;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.18); max-width: 320px;
  }
  #tooltip .t-name { font-weight: 600; margin-bottom: 3px; word-break: break-word; }
  #tooltip .t-row { color: var(--ink-2); }
  aside {
    flex: 0 1 340px; min-width: 280px; border-left: 1px solid var(--border);
    padding: 20px 20px; max-height: calc(100vh - 61px); overflow-y: auto;
  }
  @media (max-width: 900px) {
    aside { flex-basis: 100%; border-left: none; border-top: 1px solid var(--border); max-height: none; }
  }
  aside h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--ink-muted); margin: 0 0 6px; }
  .findings-hint { font-size: 11.5px; color: var(--ink-muted); margin: 0 0 12px; }
  .finding { border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 10px; cursor: pointer; }
  .finding.selected { border-color: var(--status-critical); box-shadow: 0 0 0 1px var(--status-critical); }
  .finding-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .sev-chip {
    font-size: 10px; text-transform: uppercase; letter-spacing: .03em; font-weight: 700;
    padding: 2px 7px; border-radius: 999px; color: #fff;
  }
  .sev-warning { background: var(--status-warning); color: #3a2a00; }
  .sev-serious { background: var(--status-serious); }
  .sev-critical { background: var(--status-critical); }
  .finding-id { font-weight: 600; font-size: 13px; }
  .finding .dollars { margin-left: auto; font-size: 13px; }
  .finding .evidence { color: var(--ink-2); font-size: 12.5px; margin-bottom: 6px; }
  .finding .fix { font-size: 12.5px; }
  .finding .fix::before { content: "Fix: "; font-weight: 600; }
  .no-findings { color: var(--ink-muted); font-size: 13px; }
  footer { padding: 14px 24px; color: var(--ink-muted); font-size: 11.5px; border-top: 1px solid var(--border); }
</style>
</head>
<body>
__UNPRICED_BANNER__
<header>
  <h1>&#9889; wattage &mdash; burn map</h1>
  <span class="source">__SOURCE__</span>
  <div class="spacer"></div>
  <div class="__SCORE_CHIP_CLASS__">__SCORE_CHIP_CONTENT__</div>
  <button class="theme-toggle" onclick="toggleTheme()">&#9788;/&#9789; theme</button>
</header>
<div class="stat-strip">
__STAT_STRIP__
</div>
<div class="breakdown-wrap">
  <div class="breakdown-bar">__BREAKDOWN_BAR__</div>
  <div class="breakdown-caption">__BREAKDOWN_CAPTION__</div>
</div>
<main>
  <div class="graph-panel">
    <div class="graph-toolbar">
      <div class="breadcrumb" id="breadcrumb"></div>
      <button class="reset-zoom" id="reset-zoom" onclick="resetView()">reset view</button>
    </div>
    <div class="graph-box" id="graph-box">
      <svg id="flamegraph" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
    <div class="legend" id="legend"></div>
  </div>
  <aside>
    <h2>Findings (__FINDING_COUNT__)</h2>
    <p class="findings-hint">__FINDINGS_HINT__</p>
    __FINDINGS_HTML__
  </aside>
</main>
<footer>pricing: __PRICING_VERSION__ &middot; generated __GENERATED_AT__</footer>
<div id="tooltip"></div>
<script>
const DATA = __TREE_JSON__;
const CATEGORY_COLORS = {
  input: 'var(--cat-input)', cache_read: 'var(--cat-cache-read)', cache_creation: 'var(--cat-cache-creation)',
  reasoning: 'var(--cat-reasoning)', output: 'var(--cat-output)', tool_io: 'var(--cat-tool-io)'
};
const CATEGORY_LABELS = {
  input: 'Input', cache_read: 'Cache read', cache_creation: 'Cache write',
  reasoning: 'Reasoning', output: 'Output', tool_io: 'Tool / retrieval I/O'
};
const BAND_HEIGHT = 26;
const svgNS = 'http://www.w3.org/2000/svg';
const svg = document.getElementById('flamegraph');
const graphBox = document.getElementById('graph-box');
const tooltip = document.getElementById('tooltip');

let zoomStack = [DATA];
let highlightSpans = null;  // Set of span ids when a finding is selected
let selectedFinding = null;

function colorFor(node) {
  if (node.kind === 'segment') return CATEGORY_COLORS[node.category] || 'var(--neutral-frame)';
  return null; // structural frame; painted a depth-based neutral shade
}

function layout(node, x, width, depth, out) {
  out.push({ node: node, x: x, width: width, depth: depth });
  if (!node.children || node.children.length === 0 || width < 0.5) return;
  const total = node.value || 1;
  let childX = x;
  for (const child of node.children) {
    const childWidth = (child.value / total) * width;
    if (childWidth > 0.15) layout(child, childX, childWidth, depth + 1, out);
    childX += childWidth;
  }
}

function fmtTokens(n) {
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return String(n);
}
function fmtDollars(n) {
  if (n === null || n === undefined) return null;
  return '$' + n.toFixed(n < 1 ? 4 : 2);
}

function render() {
  hideTooltip();
  // Pixel-space layout: frames are laid out at the container's real width
  // (re-run on resize), so the graph fills its panel edge-to-edge at crisp
  // 11px labels — never letterboxed or uniformly shrunk by viewBox scaling.
  const pixelWidth = Math.max(graphBox.clientWidth, 320);
  const root = zoomStack[zoomStack.length - 1];
  const rects = [];
  layout(root, 0, pixelWidth, 0, rects);
  const maxDepth = rects.reduce((m, r) => Math.max(m, r.depth), 0);
  const height = (maxDepth + 1) * BAND_HEIGHT;
  svg.setAttribute('width', pixelWidth);
  svg.setAttribute('height', height);
  svg.innerHTML = '';

  for (const r of rects) {
    const isSegment = r.node.kind === 'segment';
    const isLeaf = !r.node.children || r.node.children.length === 0;
    const fill = isSegment ? colorFor(r.node) : (r.depth % 2 === 0 ? 'var(--neutral-frame)' : 'var(--neutral-frame-2)');
    const rect = document.createElementNS(svgNS, 'rect');
    rect.setAttribute('x', r.x.toFixed(2));
    rect.setAttribute('y', r.depth * BAND_HEIGHT);
    rect.setAttribute('width', Math.max(r.width - 0.6, 0).toFixed(2));
    rect.setAttribute('height', BAND_HEIGHT - 2);
    rect.setAttribute('fill', fill);
    let cls = 'frame' + (isLeaf ? ' leaf' : '');
    if (highlightSpans !== null) {
      cls += highlightSpans.has(r.node.span_id) ? ' hot' : (spanInSubtree(r.node) ? '' : ' dimmed');
    }
    rect.setAttribute('class', cls);
    if (!isLeaf) {
      rect.addEventListener('click', () => { zoomStack.push(r.node); render(); });
    }
    rect.addEventListener('mousemove', (e) => showTooltip(e, r.node));
    rect.addEventListener('mouseleave', hideTooltip);
    svg.appendChild(rect);

    if (r.width > 16) {
      const text = document.createElementNS(svgNS, 'text');
      text.setAttribute('x', (r.x + 4).toFixed(2));
      text.setAttribute('y', r.depth * BAND_HEIGHT + BAND_HEIGHT / 2 + 4);
      let labelCls = 'frame-label' + (isSegment ? ' on-color' : '');
      if (highlightSpans !== null && !highlightSpans.has(r.node.span_id) && !spanInSubtree(r.node)) {
        labelCls += ' dimmed';
      }
      text.setAttribute('class', labelCls);
      text.style.pointerEvents = 'none';
      text.textContent = r.node.name;
      svg.appendChild(text);
      // Truncate against the box's *actual* rendered text width (not a
      // fixed char-width guess) so long labels never overflow into
      // neighboring frames or past the container's right edge.
      const available = r.width - 8;
      let label = r.node.name;
      while (label.length > 1 && text.getComputedTextLength() > available) {
        label = label.slice(0, -1);
        text.textContent = label + '…';
      }
      if (text.getComputedTextLength() > available) svg.removeChild(text);
    }
  }
  renderBreadcrumb();
}

const subtreeCache = new Map();
function spanInSubtree(node) {
  // A structural ancestor stays undimmed when any highlighted span lives
  // beneath it, so the path to the hot frames remains readable.
  if (highlightSpans === null) return true;
  if (subtreeCache.has(node)) return subtreeCache.get(node);
  let hit = highlightSpans.has(node.span_id);
  if (!hit && node.children) {
    for (const child of node.children) {
      if (spanInSubtree(child)) { hit = true; break; }
    }
  }
  subtreeCache.set(node, hit);
  return hit;
}

function renderBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  el.innerHTML = '';
  zoomStack.forEach((node, i) => {
    if (i > 0) {
      const sep = document.createElement('span');
      sep.className = 'sep'; sep.textContent = '/';
      el.appendChild(sep);
    }
    const span = document.createElement('span');
    span.textContent = node.name;
    span.addEventListener('click', () => { zoomStack = zoomStack.slice(0, i + 1); render(); });
    el.appendChild(span);
  });
}

function resetView() {
  zoomStack = [DATA];
  clearHighlight();
  render();
}

function clearHighlight() {
  highlightSpans = null;
  selectedFinding = null;
  subtreeCache.clear();
  document.querySelectorAll('.finding.selected').forEach((el) => el.classList.remove('selected'));
}

function toggleFinding(el) {
  // Clicking a finding lights up exactly the frames whose spans burned that
  // money; clicking it again (or reset view) clears the highlight.
  let spans = [];
  try { spans = JSON.parse(el.getAttribute('data-span-ids') || '[]'); } catch (e) { spans = []; }
  if (selectedFinding === el || spans.length === 0) { clearHighlight(); render(); return; }
  clearHighlight();
  selectedFinding = el;
  el.classList.add('selected');
  highlightSpans = new Set(spans);
  subtreeCache.clear();
  render();
}

function showTooltip(evt, node) {
  const dollars = fmtDollars(node.dollars);
  let html = '<div class="t-name">' + escapeHtml(node.name) + '</div>';
  html += '<div class="t-row">' + node.kind + ' &middot; ' + fmtTokens(node.tokens) + ' tokens';
  if (dollars !== null) html += ' &middot; ' + dollars;
  html += '</div>';
  if (node.model) html += '<div class="t-row">model: ' + escapeHtml(node.model) + '</div>';
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  const x = evt.clientX + 14, y = evt.clientY + 14;
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}
function hideTooltip() { tooltip.style.display = 'none'; }
function escapeHtml(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}

function renderLegend() {
  const el = document.getElementById('legend');
  el.innerHTML = '';
  for (const key of Object.keys(CATEGORY_LABELS)) {
    const item = document.createElement('span');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = CATEGORY_COLORS[key];
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(CATEGORY_LABELS[key]));
    el.appendChild(item);
  }
}

function toggleTheme() {
  const root = document.documentElement;
  const current = root.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('wattage-theme', next); } catch (e) { /* private mode etc. */ }
}
try {
  const saved = localStorage.getItem('wattage-theme');
  if (saved === 'dark' || saved === 'light') document.documentElement.setAttribute('data-theme', saved);
} catch (e) { /* storage unavailable; system preference applies */ }

document.querySelectorAll('.finding[data-span-ids]').forEach((el) => {
  el.addEventListener('click', () => toggleFinding(el));
});

if (typeof ResizeObserver !== 'undefined') {
  let lastWidth = 0;
  new ResizeObserver(() => {
    const w = graphBox.clientWidth;
    if (Math.abs(w - lastWidth) > 1) { lastWidth = w; render(); }
  }).observe(graphBox);
}

render();
renderLegend();
</script>
</body>
</html>
"""
)


def _score_chip(report: Report) -> tuple[str, str]:
    """(css_class, inner_html) for the header's score pill -- a grade
    computed from an incomplete cost figure is exactly the kind of
    plausible-looking-but-wrong number this project exists to avoid, so an
    unpriced trace gets a distinct warning pill instead of a letter grade."""
    if report.unpriced_calls:
        call_word = "call" if report.unpriced_calls == 1 else "calls"
        content = f"<b>&#9888; unpriced</b> ({report.unpriced_calls} {escape(call_word)})"
        return "score-chip unpriced", content
    score = report.score
    grade_class = _GRADE_CLASS.get(score.grade, "grade-bad")
    if score.recoverable_dollars > 0:
        detail = f"{escape(format_dollars(score.recoverable_dollars))} recoverable"
    else:
        detail = "efficient"
    content = f"<b>{escape(score.grade)} ({score.efficiency})</b> &middot; {detail}"
    return f"score-chip {grade_class}", content


def _unpriced_banner(report: Report) -> str:
    if not report.unpriced_calls:
        return ""
    call_word = "call" if report.unpriced_calls == 1 else "calls"
    models = escape(", ".join(report.unpriced_models))
    return (
        '<div class="unpriced-banner">&#9888; '
        f"{report.unpriced_calls} {call_word} could not be priced (no registry entry for "
        f"{models}) &mdash; total cost and score are incomplete.</div>"
    )


def _stat(label: str, value: str, sub: str = "", value_class: str = "") -> str:
    cls = f"stat-value{f' {value_class}' if value_class else ''}"
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return (
        '<div class="stat">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="{cls} num">{value}</div>'
        f"{sub_html}</div>"
    )


def _stat_strip(report: Report) -> str:
    """The tweetable summary card: the numbers a reader screenshots."""
    breakdown = report.token_breakdown
    total_tokens = sum(breakdown.values())
    prompt_side = breakdown.get("input", 0) + breakdown.get("cache_read", 0)
    cache_rate = breakdown.get("cache_read", 0) / prompt_side if prompt_side else 0.0
    stats = [
        _stat("Total cost", escape(format_dollars(report.total_dollars))),
        _stat("Total tokens", f"{total_tokens:,}"),
        _stat(
            "Cache hit rate",
            f"{cache_rate:.0%}",
            sub="of prompt-side tokens read from cache",
        ),
    ]
    measured = report.score.recoverable_dollars
    if measured > 0:
        stats.append(
            _stat(
                "Measured waste",
                escape(format_dollars(measured)),
                sub="counts toward the grade",
                value_class="money-bad",
            )
        )
    estimated = report.score.estimated_extra_dollars
    if estimated > 0:
        stats.append(
            _stat(
                "Estimated findings",
                f"~{escape(format_dollars(estimated))}",
                sub="heuristic; never graded",
            )
        )
    if report.score.monthly_projection is not None:
        stats.append(
            _stat(
                "Monthly projection",
                escape(format_dollars(report.score.monthly_projection)),
                sub="at your stated run rate",
            )
        )
    return "\n".join(stats)


_BAR_ORDER = ("input", "cache_read", "cache_creation", "reasoning", "output")
_BAR_COLORS = {
    "input": "var(--cat-input)",
    "cache_read": "var(--cat-cache-read)",
    "cache_creation": "var(--cat-cache-creation)",
    "reasoning": "var(--cat-reasoning)",
    "output": "var(--cat-output)",
}
_BAR_LABELS = {
    "input": "input",
    "cache_read": "cache read",
    "cache_creation": "cache write",
    "reasoning": "reasoning",
    "output": "output",
}


def _breakdown_bar(report: Report) -> tuple[str, str]:
    breakdown = report.token_breakdown
    total = sum(breakdown.values())
    if total <= 0:
        return "", ""
    spans = []
    caption_parts = []
    for key in _BAR_ORDER:
        tokens = breakdown.get(key, 0)
        if tokens <= 0:
            continue
        pct = tokens / total * 100
        spans.append(
            f'<span style="width:{pct:.2f}%;background:{_BAR_COLORS[key]}" '
            f'title="{_BAR_LABELS[key]}: {tokens:,} tokens ({pct:.1f}%)"></span>'
        )
        caption_parts.append(f"{_BAR_LABELS[key]} {pct:.1f}%")
    return "".join(spans), " &middot; ".join(caption_parts)


def _json_for_script(data: object) -> str:
    """json.dumps(), hardened for embedding inside a <script> block.

    Tree node names come straight from trace content (tool names, model
    strings, retrieval queries) -- untrusted input, since anyone can hand
    `wattage report --html` an arbitrary trace file. json.dumps() escapes
    for JSON string syntax but not for HTML: a literal `</script>` in a
    string value closes the tag early and lets whatever follows execute as
    a new <script> in the shareable, often-opened-in-a-browser HTML report.
    Escaping </>/& as \\u-sequences (same approach as Django's json_script)
    is a no-op for JSON.parse but can no longer terminate the tag; U+2028/
    U+2029 are escaped too since they're valid JSON but were historically
    invalid unescaped in a JS string literal.
    """
    encoded = json.dumps(data)
    return (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(" ", "\\u2028")
        .replace(" ", "\\u2029")
    )


_PLACEHOLDER_RE = re.compile(
    "|".join(
        re.escape(p)
        for p in (
            "__TITLE__",
            "__SOURCE__",
            "__UNPRICED_BANNER__",
            "__SCORE_CHIP_CLASS__",
            "__SCORE_CHIP_CONTENT__",
            "__STAT_STRIP__",
            "__BREAKDOWN_BAR__",
            "__BREAKDOWN_CAPTION__",
            "__FINDING_COUNT__",
            "__FINDINGS_HINT__",
            "__FINDINGS_HTML__",
            "__PRICING_VERSION__",
            "__GENERATED_AT__",
            "__TREE_JSON__",
        )
    )
)


def render_html(trace: Trace, report: Report) -> str:
    tree = build_tree(trace)
    findings_html = _render_findings(report)
    score_chip_class, score_chip_content = _score_chip(report)
    bar_html, bar_caption = _breakdown_bar(report)
    findings_hint = (
        "Click a finding to light up the frames that burned it." if report.findings else ""
    )

    # Trace content (finding evidence/fix text, tool/model/query names) can
    # legitimately contain the literal text of another placeholder token --
    # e.g. a tool named "__TREE_JSON__" -- so sequential str.replace() calls
    # would let a later call re-match text a prior call just inserted,
    # substituting again inside already-substituted content. A single regex
    # pass over the original, static _TEMPLATE never rescans replacement
    # text, so this can't happen regardless of what's in the substitutions.
    substitutions = {
        "__TITLE__": escape(f"wattage burn map — {report.trace_source}"),
        "__SOURCE__": escape(report.trace_source),
        "__UNPRICED_BANNER__": _unpriced_banner(report),
        "__SCORE_CHIP_CLASS__": score_chip_class,
        "__SCORE_CHIP_CONTENT__": score_chip_content,
        "__STAT_STRIP__": _stat_strip(report),
        "__BREAKDOWN_BAR__": bar_html,
        "__BREAKDOWN_CAPTION__": bar_caption,
        "__FINDING_COUNT__": str(len(report.findings)),
        "__FINDINGS_HINT__": findings_hint,
        "__FINDINGS_HTML__": findings_html,
        "__PRICING_VERSION__": escape(report.pricing_version),
        "__GENERATED_AT__": escape(report.generated_at),
        "__TREE_JSON__": _json_for_script(tree),
    }
    return _PLACEHOLDER_RE.sub(lambda m: substitutions[m.group(0)], _TEMPLATE)


def _render_findings(report: Report) -> str:
    if not report.findings and report.unpriced_calls:
        return (
            '<p class="no-findings">No findings on the priced portion of this trace — '
            f"{report.unpriced_calls} call(s) could not be priced, so this is not a "
            "confirmed-efficient trace.</p>"
        )
    if not report.findings:
        return '<p class="no-findings">No findings — this trace looks efficient.</p>'

    parts = []
    for finding in report.findings:
        status = _SEVERITY_STATUS.get(finding.severity.value, "warning")
        # JSON-encoded id list: span ids are arbitrary trace strings (the
        # OTLP adapter accepts any non-empty spanId), so a delimiter-joined
        # form could split wrongly and silently break the click-to-highlight
        # wiring. JSON round-trips any id exactly; attribute-escaped so
        # arbitrary trace content stays inert.
        span_ids_attr = escape(json.dumps(finding.span_ids), {'"': "&quot;"})
        dollars = format_dollars(finding.wasted_dollars)
        if finding.basis == FindingBasis.estimated:
            dollars = f"~{dollars}"
        parts.append(
            f'<div class="finding" data-span-ids="{span_ids_attr}">'
            '<div class="finding-head">'
            f'<span class="sev-chip sev-{status}">{escape(finding.severity.value)}</span>'
            f'<span class="finding-id">{escape(finding.id)}</span>'
            f'<span class="dollars num">{escape(dollars)}</span>'
            "</div>"
            f'<div class="evidence">{escape(finding.evidence)}</div>'
            f'<div class="fix">{escape(finding.fix)}</div>'
            "</div>"
        )
    return "\n".join(parts)
