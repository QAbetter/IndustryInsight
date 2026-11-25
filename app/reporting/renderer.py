"""Markdown 报告 -> 自包含 HTML（用于下载、归档与演示）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape


def render_report_html(
    report_md: str,
    meta: dict[str, Any],
    metrics: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    body_html = markdown.markdown(
        report_md,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
        output_format="html5",
    )
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report_template.html")

    rows = []
    for stage, values in (metrics.get("by_stage") or {}).items():
        rows.append(
            {
                "stage": stage,
                "calls": values.get("llm_calls", 0),
                "tokens": values.get("prompt_tokens", 0) + values.get("completion_tokens", 0),
                "latency": round(float(values.get("latency_ms", 0)), 1),
            }
        )
    rows.sort(key=lambda row: row["latency"], reverse=True)

    return template.render(
        topic=meta.get("topic", ""),
        run_id=meta.get("run_id", ""),
        generated_at=meta.get("generated_at", ""),
        elapsed_s=meta.get("elapsed_s", 0),
        chars=meta.get("chars", 0),
        citations=meta.get("citations", 0),
        evidence_count=meta.get("evidence_count", 0),
        provider=meta.get("provider", ""),
        model=meta.get("model", ""),
        top_k=meta.get("top_k", ""),
        threshold=meta.get("threshold", ""),
        body_html=body_html,
        stage_rows=rows,
        total_tokens=metrics.get("tokens", {}).get("total", 0),
        cost_usd=metrics.get("cost_usd", 0),
        llm_calls=metrics.get("llm_calls", 0),
        tool_calls=metrics.get("tool_calls", 0),
        evidence_items=(evidence or {}).get("items", [])[:12],
    )
