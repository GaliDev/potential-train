"""Render the benchmark comparison as a single self-contained HTML page.

No JS dependencies - tables plus CSS bar charts, openable straight from disk.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

# Columns shown in the ranking table: (key in metrics dict, label, format, higher_is_better)
_METRIC_COLS = [
    ("pass_accuracy", "Pass accuracy", "{:.0%}", True),
    ("cohen_kappa", "Cohen's κ", "{:.3f}", True),
    ("spearman", "Spearman ρ", "{:.3f}", True),
    ("score_mae", "Score MAE", "{:.3f}", False),
    ("avg_latency_s", "Latency/item (s)", "{:.1f}", False),
    ("total_cost_usd", "Est. cost ($)", "{:.4f}", False),
]

_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 2rem auto;
       max-width: 1000px; color: #1a1d27; padding: 0 1rem; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.15rem; margin-top: 2.2rem; }
.meta { color: #666; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; margin-top: .8rem; font-size: .92rem; }
th, td { border: 1px solid #e2e4ea; padding: .45rem .6rem; text-align: left; }
th { background: #f4f5f8; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { background: #e7f6ec; font-weight: 600; }
.bar-wrap { background: #eef0f4; border-radius: 4px; height: 16px; min-width: 160px; }
.bar { background: #4c7dd0; border-radius: 4px; height: 16px; }
.note { background: #fff8e6; border: 1px solid #f0e0b0; border-radius: 6px;
        padding: .7rem 1rem; font-size: .9rem; margin-top: 1rem; }
.target-met { color: #15803d; font-weight: 600; } .target-miss { color: #b91c1c; }
footer { margin-top: 2.5rem; color: #888; font-size: .8rem; }
"""


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "–"
    return fmt.format(value)


def _best_value(rows: list[dict], key: str, higher_is_better: bool):
    vals = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
    if not vals:
        return None
    return max(vals) if higher_is_better else min(vals)


def render_report(payload: dict) -> str:
    """payload: output of run_benchmark.build_payload()."""
    rows = payload["models"]
    e = html.escape

    ranking_rows = []
    for row in rows:
        cells = [f"<td><b>{e(row['display_name'])}</b><br><span class='meta'>{e(row['model_id'])}</span></td>"]
        for key, _label, fmt, higher in _METRIC_COLS:
            val = row["metrics"].get(key)
            best = _best_value(rows, key, higher)
            cls = "num best" if val is not None and val == best and len(rows) > 1 else "num"
            cells.append(f"<td class='{cls}'>{_fmt(val, fmt)}</td>")
        cells.append(f"<td class='num'>{row['errors']}</td>")
        ranking_rows.append("<tr>" + "".join(cells) + "</tr>")

    kappa_bars = []
    for row in rows:
        kappa = row["metrics"].get("cohen_kappa") or 0.0
        width = max(0.0, min(1.0, kappa)) * 100
        kappa_bars.append(
            f"<tr><td>{e(row['display_name'])}</td>"
            f"<td><div class='bar-wrap'><div class='bar' style='width:{width:.0f}%'></div></div></td>"
            f"<td class='num'>{kappa:.3f}</td></tr>"
        )

    criteria = payload["criteria"]
    crit_header = "".join(f"<th>{e(c)}</th>" for c in criteria)
    crit_rows = []
    for row in rows:
        tds = "".join(
            f"<td class='num'>{_fmt(row['per_criterion_mean'].get(c), '{:.2f}')}</td>"
            for c in criteria
        )
        crit_rows.append(f"<tr><td>{e(row['display_name'])}</td>{tds}</tr>")

    parse_rows = []
    for row in rows:
        pc = row.get("parse_counts", {})
        total = sum(pc.values()) or 1
        parse_rows.append(
            f"<tr><td>{e(row['display_name'])}</td>"
            f"<td class='num'>{pc.get('json', 0) / total:.0%}</td>"
            f"<td class='num'>{pc.get('regex', 0)}</td>"
            f"<td class='num'>{pc.get('fallback', 0)}</td></tr>"
        )

    target_notes = []
    for row in rows:
        acc = row["metrics"].get("pass_accuracy")
        kappa = row["metrics"].get("cohen_kappa")
        ok = acc is not None and kappa is not None and acc >= 0.8 and kappa >= 0.6
        cls = "target-met" if ok else "target-miss"
        target_notes.append(
            f"<li><span class='{cls}'>{e(row['display_name'])}: "
            f"{'meets' if ok else 'does not meet'}</span> the platform bar "
            f"(accuracy {_fmt(acc, '{:.0%}')}, κ {_fmt(kappa, '{:.3f}')})</li>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    task_mix = ", ".join(f"{k}: {v}" for k, v in payload["task_mix"].items())

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>HF LLM-as-Judge Benchmark</title><style>{_CSS}</style></head>
<body>
<h1>HF LLM-as-Judge Benchmark</h1>
<p class="meta">Generated {generated} · {payload['n_items']} gold items ({task_mix})
· judged on 5 criteria: correctness, faithfulness, completeness, coherence, safety</p>

<h2>Ranking — agreement with human gold labels</h2>
<table>
<tr><th>Model</th>{"".join(f"<th>{label}</th>" for _, label, _, _ in _METRIC_COLS)}<th>Errors</th></tr>
{"".join(ranking_rows)}
</table>
<p class="meta">Green = best in column. Pass accuracy / κ / ρ: higher is better. MAE, latency, cost: lower is better.</p>

<h2>Cohen's κ (chance-corrected agreement)</h2>
<table><tr><th>Model</th><th>κ</th><th class="num">value</th></tr>{"".join(kappa_bars)}</table>

<h2>Mean score per criterion</h2>
<table><tr><th>Model</th>{crit_header}</tr>{"".join(crit_rows)}</table>
<p class="meta">A model whose row sits notably above the others is a lenient judge; below, a harsh one.</p>

<h2>Output reliability (structured-output discipline)</h2>
<table><tr><th>Model</th><th>Clean JSON</th><th>Regex rescue</th><th>Fallback</th></tr>{"".join(parse_rows)}</table>
<p class="meta">Fallback verdicts default toward score 3 and add noise — prefer models with high clean-JSON rates.</p>

<h2>Platform bar (&ge;80% agreement, κ &ge; 0.6)</h2>
<ul>{"".join(target_notes)}</ul>

<div class="note"><b>Reading this report:</b> agreement is measured on the project's
hand-labeled gold set against <code>gold_pass</code> / <code>gold_score</code> using the
same metrics module as the GPT-4o panel benchmark. Costs are indicative estimates from
serverless per-token rates.</div>

<footer>hf_judges benchmark · branch llm-as-judges</footer>
</body></html>
"""
