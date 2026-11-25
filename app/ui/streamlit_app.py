"""IndustryInsight Streamlit 前端：报告生成、知识库管理、运行历史与指标观察。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.config import Settings
from app.pipeline.runner import PipelineResult, ResearchPipeline
from app.rag.knowledge import KnowledgeBase

PROVIDER_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
}

st.set_page_config(
    page_title="IndustryInsight · 多 Agent 行业调研",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
<style>
  .block-container { padding-top: 2.2rem; }
  .hero {
    background: linear-gradient(120deg, #0f1f3d 0%, #1f3f7a 55%, #1f5eff 100%);
    color: #fff; border-radius: 16px; padding: 26px 30px; margin-bottom: 18px;
  }
  .hero h1 { margin: 0 0 6px; font-size: 26px; }
  .hero p { margin: 0; opacity: .82; font-size: 13px; }
  .stButton > button[kind="primary"] { width: 100%; }
  div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e3e8ef; border-radius: 12px; padding: 12px 14px;
  }
</style>
""",
    unsafe_allow_html=True,
)


def _load_settings() -> Settings:
    return Settings.from_file(ROOT / "config" / "settings.yaml")


def _format_event(event: dict) -> str:
    kind = event.get("type")
    if kind == "task_start":
        return f">> {event.get('task_id')} · {event.get('title')} ({event.get('agent')})"
    if kind == "task_end":
        flag = "完成" if event.get("status") == "ok" else "失败"
        return f"   [{flag}] {event.get('task_id')} {event.get('duration_ms', 0) / 1000:.2f}s"
    if kind == "tool_call":
        return (
            f"   [工具] {event.get('tool')} "
            f"{'成功' if event.get('ok') else '失败'} {event.get('duration_ms', 0):.0f}ms"
        )
    if kind == "coverage_check":
        return (
            f"   [质检] 第 {event.get('round')} 轮："
            f"{'证据充分' if event.get('sufficient') else '触发补充检索'}"
        )
    if kind == "validation_failed":
        return f"   [校验] {event.get('stage')} 输出不合规，正在重试"
    if kind == "kb_ingest":
        return f"   [知识库] 新增 {event.get('chunks_added')} 个片段"
    if kind == "kb_ready":
        return f"   [知识库] {event.get('chunks')} 个片段 / {event.get('embedder')}"
    return ""


def _render_result(result: PipelineResult) -> None:
    metrics = result.metrics or {}
    tokens = metrics.get("tokens", {})
    st.success(f"报告生成完成：{len(result.report):,} 字 · {result.elapsed_ms / 1000:.1f}s · {result.run_dir}")
    columns = st.columns(5)
    columns[0].metric("正文字数", f"{len(result.report):,}")
    columns[1].metric("证据条目", f"{len((result.evidence or {}).get('items', []))}")
    columns[2].metric("Token", f"{tokens.get('total', 0):,}")
    columns[3].metric("LLM 调用", f"{metrics.get('llm_calls', 0)}")
    columns[4].metric("预估成本", f"${metrics.get('cost_usd', 0):.4f}")

    tabs = st.tabs(["报告", "结构化分析", "证据链", "运行指标", "事件日志"])
    with tabs[0]:
        st.markdown(result.report)
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "下载 Markdown",
            result.report.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        html_path = result.artifacts.get("report_html")
        if html_path and Path(html_path).exists():
            download_columns[1].download_button(
                "下载 HTML",
                Path(html_path).read_bytes(),
                file_name="report.html",
                mime="text/html",
                use_container_width=True,
            )
        if metrics:
            download_columns[2].download_button(
                "下载指标 JSON",
                json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="metrics.json",
                mime="application/json",
                use_container_width=True,
            )
    with tabs[1]:
        if result.analysis:
            st.json(result.analysis)
        else:
            st.info("没有可展示的分析结果")
    with tabs[2]:
        items = (result.evidence or {}).get("items", [])
        if items:
            frame = pd.DataFrame(
                [
                    {
                        "编号": item.get("id"),
                        "事实": item.get("claim"),
                        "类型": item.get("source_type"),
                        "来源": item.get("source") or item.get("url"),
                        "页码": item.get("page"),
                        "置信度": item.get("confidence"),
                    }
                    for item in items
                ]
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("本次运行没有证据条目")
    with tabs[3]:
        by_agent = metrics.get("by_agent") or {}
        if by_agent:
            agent_rows = []
            for agent, values in by_agent.items():
                agent_rows.append(
                    {
                        "Agent": agent,
                        "Token": values.get("prompt_tokens", 0) + values.get("completion_tokens", 0),
                        "LLM 调用": values.get("llm_calls", 0),
                        "工具调用": values.get("tool_calls", 0),
                        "耗时(ms)": values.get("task_duration_ms", 0),
                    }
                )
            st.dataframe(pd.DataFrame(agent_rows), use_container_width=True, hide_index=True)
        stage_rows = []
        for stage, values in (metrics.get("by_stage") or {}).items():
            stage_rows.append(
                {
                    "阶段": stage,
                    "Token": values.get("prompt_tokens", 0) + values.get("completion_tokens", 0),
                    "LLM 调用": values.get("llm_calls", 0),
                    "耗时(ms)": values.get("latency_ms", 0),
                }
            )
        if stage_rows:
            stage_frame = pd.DataFrame(stage_rows).sort_values("耗时(ms)", ascending=False)
            st.bar_chart(stage_frame.set_index("阶段")["耗时(ms)"])
            st.dataframe(stage_frame, use_container_width=True, hide_index=True)
    with tabs[4]:
        events_path = result.run_dir / "events.jsonl"
        if events_path.exists():
            lines = events_path.read_text(encoding="utf-8").splitlines()
            st.code("\n".join(lines[-80:]), language="json")
        else:
            st.info("没有事件日志")


def _render_history(settings: Settings) -> None:
    runs_dir = settings.path(settings.pipeline.output_dir)
    if not runs_dir.exists():
        st.info("暂无历史运行")
        return
    run_dirs = sorted(
        [path for path in runs_dir.iterdir() if path.is_dir() and (path / "report.md").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        st.info("暂无历史运行")
        return
    selected = st.selectbox("选择运行记录", [path.name for path in run_dirs], index=0)
    run_dir = runs_dir / selected
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    columns = st.columns([2, 1])
    with columns[0]:
        st.markdown(report)
    with columns[1]:
        st.download_button(
            "下载历史报告",
            report.encode("utf-8"),
            file_name=f"{selected}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.metric("总 Token", f"{metrics.get('tokens', {}).get('total', 0):,}")
            st.metric("LLM 调用", metrics.get("llm_calls", 0))
            st.metric("工具调用", metrics.get("tool_calls", 0))
            st.json(metrics.get("slowest_stages", []))


def _render_kb(settings: Settings) -> None:
    try:
        kb = KnowledgeBase(settings)
    except Exception as exc:  # noqa: BLE001
        st.error(f"知识库初始化失败：{exc}")
        return
    stats = kb.stats()
    columns = st.columns(5)
    columns[0].metric("状态", "异常" if stats["load_error"] else "正常")
    columns[1].metric("向量片段", f"{stats['chunks']:,}")
    columns[2].metric("文档数", stats["sources"])
    columns[3].metric("向量后端", stats["backend"])
    columns[4].metric("Embedding", stats["embedder"])
    if stats["load_error"]:
        st.warning(stats["load_error"])

    sources = kb.list_sources()
    if sources:
        st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

    st.subheader("增量导入")
    uploads = st.file_uploader(
        "上传 PDF / Word / Markdown / TXT",
        type=["pdf", "docx", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if uploads and st.button("导入到知识库", use_container_width=True):
        upload_dir = settings.path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for upload in uploads:
            path = upload_dir / upload.name
            path.write_bytes(upload.getbuffer())
            paths.append(str(path))
        with st.spinner("解析、切片与向量化中..."):
            ingest_stats = kb.ingest(paths)
        st.success(
            f"导入完成：新增 {ingest_stats.chunks_added} 个片段，"
            f"跳过 {ingest_stats.chunks_skipped} 个"
        )
        st.rerun()

    if st.button("用 data/raw 重建索引", use_container_width=True):
        from app.sample_data import ensure_demo_corpus

        ensure_demo_corpus(settings.root)
        with st.spinner("重建索引中..."):
            ingest_stats = kb.ingest([settings.path("data/raw")], reset=True)
        st.success(f"重建完成：{ingest_stats.chunks_added} 个片段")
        st.rerun()


settings = _load_settings()

st.markdown(
    """
<div class="hero">
  <h1>IndustryInsight · 多 Agent 行业调研</h1>
  <p>研究员检索 → 证据合并 → 分析师结构化分析 → 撰写员长文成稿，RAG 与联网信息融合，全链路可溯源。</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("研究任务")
    topic = st.text_input("研究主题", value="中国新能源汽车行业（2024）")
    focus = st.text_area(
        "关注点（可选）",
        placeholder="例如：重点关注竞争格局、价格战与海外市场",
        height=80,
    )
    target_chars = st.slider("目标字数", 3000, 12000, settings.pipeline.target_chars, step=1000)

    st.divider()
    st.subheader("运行模式")
    mode = st.radio("模型", ["离线演示（Mock）", "真实模型"], horizontal=True)
    provider = "mock"
    model = ""
    api_key = ""
    if mode == "真实模型":
        provider = st.selectbox("Provider", list(PROVIDER_ENV), index=0)
        model = st.text_input("模型名（留空用预设）", value="")
        api_key = st.text_input(
            f"{PROVIDER_ENV[provider]}（留空则读取 .env）",
            type="password",
        )
        st.caption("也可在 .env 中配置；Key 仅保存在当前会话内存。")
    online = st.checkbox("启用联网检索（Tavily / DuckDuckGo）", value=False)
    st.caption("离线模式使用 fixtures 与内置知识库，不需要任何 API Key。")

    st.divider()
    st.subheader("RAG 参数")
    top_k = st.slider("Top-K", 1, 10, settings.rag.top_k)
    threshold = st.slider("相似度阈值", 0.0, 0.6, float(settings.rag.score_threshold), step=0.02)

    run_clicked = st.button("生成报告", type="primary", use_container_width=True)

main_tabs = st.tabs(["生成", "知识库", "历史运行"])

with main_tabs[0]:
    result = st.session_state.get("result")
    if run_clicked:
        if not topic.strip():
            st.error("请输入研究主题")
        else:
            try:
                if api_key.strip() and mode == "真实模型":
                    os.environ[PROVIDER_ENV[provider]] = api_key.strip()
                overrides = {
                    "llm": {
                        "provider": "mock" if mode == "离线演示（Mock）" else provider,
                        "model": "" if mode == "离线演示（Mock）" else model.strip(),
                    },
                    "rag": {"top_k": top_k, "score_threshold": threshold},
                    "pipeline": {"target_chars": target_chars},
                }
                if online:
                    overrides["search"] = {"provider": "auto"}
                    overrides["fetch"] = {"offline": False}
                else:
                    overrides["search"] = {"provider": "fixture"}
                    overrides["fetch"] = {"offline": True}
                run_settings = settings.with_overrides(**overrides)

                progress = st.empty()
                events: list[str] = []

                def on_event(event: dict) -> None:
                    line = _format_event(event)
                    if line:
                        events.append(line)
                        progress.code("\n".join(events[-12:]), language="text")

                pipeline = ResearchPipeline(run_settings, on_event=on_event)
                with st.spinner("多 Agent 正在协作..."):
                    st.session_state["result"] = pipeline.run(
                        topic=topic, focus=focus, target_chars=target_chars
                    )
                progress.empty()
            except Exception as exc:  # noqa: BLE001
                st.error(f"运行失败：{exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("在左侧填写研究主题并点击“生成报告”。离线演示无需 API Key，约数秒完成。")
        st.markdown(
            """
**流水线**：研究规划 → 联网检索 / 私有研报检索（并行）→ 证据合并 → 结构化分析 → 分章节长文写作。

**可观察项**：每个 Agent 的 Token、耗时、工具调用、校验重试、补搜轮次，都会写入运行目录的 `events.jsonl` 与 `metrics.json`。
"""
        )
    else:
        _render_result(result)

with main_tabs[1]:
    _render_kb(settings)

with main_tabs[2]:
    _render_history(settings)
