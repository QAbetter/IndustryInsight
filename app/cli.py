"""命令行入口：bootstrap / ingest / run / demo / eval / ui。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from app.config import ConfigError, Settings
from app.core.llm import create_llm
from app.core.logging_utils import setup_logging
from app.pipeline.evaluator import evaluate_qa, evaluate_retrieval, load_qa
from app.pipeline.runner import PipelineError, PipelineResult, ResearchPipeline
from app.rag.knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parent.parent


class ConsoleProgress:
    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "task_start":
            print(f"[task] {event.get('task_id')} · {event.get('title')} ({event.get('agent')})")
        elif kind == "task_end":
            status = "done" if event.get("status") == "ok" else "FAIL"
            print(f"[{status}] {event.get('task_id')} · {event.get('duration_ms', 0) / 1000:.2f}s")
        elif kind == "tool_call":
            flag = "ok" if event.get("ok") else "err"
            print(f"       [tool] {event.get('tool')} ({flag}, {event.get('duration_ms', 0):.0f}ms)")
        elif kind == "coverage_check":
            decision = "sufficient" if event.get("sufficient") else "insufficient"
            print(f"       [coverage] round {event.get('round')} -> {decision}")
        elif kind == "validation_failed":
            errors = "；".join(event.get("errors", [])[:2])
            print(f"       [retry] {event.get('stage')} 校验失败：{errors}")
        elif kind == "task_retry":
            print(f"       [retry] {event.get('task_id')} 第 {event.get('attempt')} 次重试")
        elif kind == "kb_ingest":
            print(
                f"[kb] 入库 {event.get('chunks_added')} 个片段，跳过 {event.get('chunks_skipped')} 个"
            )
        elif kind == "kb_ready":
            print(f"[kb] {event.get('chunks')} 个片段 / {event.get('embedder')}")
        elif kind == "run_summary":
            print(f"[run] {event.get('status')} · {event.get('elapsed_ms', 0) / 1000:.1f}s")


def _build_settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_file(ROOT / "config" / "settings.yaml")
    llm_overrides: dict[str, Any] = {}
    if getattr(args, "provider", None):
        llm_overrides["provider"] = args.provider
        llm_overrides["model"] = getattr(args, "model", "") or ""
    elif getattr(args, "model", None):
        llm_overrides["model"] = args.model

    thinking = getattr(args, "thinking", None)
    if thinking:
        if thinking == "disabled":
            llm_overrides["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            llm_overrides["extra_body"] = {
                "thinking": {"type": "enabled", "effort": thinking},
                "reasoning_effort": thinking,
            }

    rag_overrides: dict[str, Any] = {}
    if getattr(args, "top_k", None):
        rag_overrides["top_k"] = int(args.top_k)
    if getattr(args, "threshold", None) is not None:
        rag_overrides["score_threshold"] = float(args.threshold)

    search_overrides: dict[str, Any] = {}
    fetch_overrides: dict[str, Any] = {}
    if getattr(args, "online", False):
        search_overrides["provider"] = "auto"
        fetch_overrides["offline"] = False

    pipeline_overrides: dict[str, Any] = {}
    if getattr(args, "target_chars", None):
        pipeline_overrides["target_chars"] = int(args.target_chars)
    if getattr(args, "output_dir", None):
        pipeline_overrides["output_dir"] = args.output_dir

    return settings.with_overrides(
        llm=llm_overrides,
        rag=rag_overrides,
        search=search_overrides,
        fetch=fetch_overrides,
        pipeline=pipeline_overrides,
    )


def _print_result(result: PipelineResult) -> None:
    metrics = result.metrics or {}
    tokens = metrics.get("tokens", {})
    print("")
    print("=" * 68)
    print(f"运行目录: {result.run_dir}")
    print(f"报告路径: {result.artifacts.get('report', '-')}")
    print(f"HTML 版本: {result.artifacts.get('report_html', '-')}")
    print(
        f"正文 {len(result.report):,} 字 · LLM 调用 {metrics.get('llm_calls', 0)} 次 · "
        f"Token {tokens.get('total', 0):,} · 约 ${metrics.get('cost_usd', 0):.4f}"
    )
    print(f"总耗时: {result.elapsed_ms / 1000:.1f}s")
    print("Agent 用量:")
    for agent, values in (metrics.get("by_agent") or {}).items():
        print(
            f"  - {agent}: tokens={values.get('prompt_tokens', 0) + values.get('completion_tokens', 0):,}, "
            f"llm={values.get('llm_calls', 0)}, tools={values.get('tool_calls', 0)}, "
            f"耗时={values.get('task_duration_ms', 0) / 1000:.2f}s"
        )
    slowest = metrics.get("slowest_stages") or []
    if slowest:
        print("耗时最高的阶段:")
        for row in slowest:
            print(
                f"  - {row['stage']}: {row['latency_ms'] / 1000:.2f}s, "
                f"{row['llm_calls']} 次调用, {row['tokens']:,} tokens"
            )
    if result.errors:
        print("错误:")
        for task_id, error in result.errors.items():
            print(f"  - {task_id}: {error}")
    print("=" * 68)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from app.sample_data import ensure_demo_corpus

    settings = _build_settings(args)
    setup_logging()
    created = ensure_demo_corpus(settings.root)
    if created:
        print(f"已生成 {len(created)} 份演示研报")
    kb = KnowledgeBase(settings)
    stats = kb.ingest([settings.path("data/raw")], reset=True)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"知识库: {kb.stats()}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    setup_logging()
    paths = args.paths or [str(settings.path("data/raw"))]
    kb = KnowledgeBase(settings)
    stats = kb.ingest(paths, reset=args.reset)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"知识库: {kb.stats()}")
    return 0


def _run_pipeline(args: argparse.Namespace, settings: Settings | None = None) -> int:
    settings = settings or _build_settings(args)
    setup_logging()
    pipeline = ResearchPipeline(settings, on_event=ConsoleProgress())
    try:
        result = pipeline.run(
            topic=args.topic,
            focus=getattr(args, "focus", "") or "",
            target_chars=args.target_chars,
        )
    except (PipelineError, ConfigError) as exc:
        print(f"\n运行失败：{exc}")
        if getattr(exc, "run_dir", None):
            print(f"运行目录: {exc.run_dir}")
        return 1
    _print_result(result)
    if settings.llm.provider == "mock":
        print("提示：当前为离线 Mock 模式，数据为演示语料。接入真实模型请配置 .env 并使用 --provider。")
    return 0 if result.ok else 1


def cmd_demo(args: argparse.Namespace) -> int:
    from app.sample_data import ensure_demo_corpus

    settings = _build_settings(args).with_overrides(
        llm={"provider": "mock", "model": ""},
        rag={
            "embedding": "hashing",
            "dim": 768,
            "persist_dir": "data/index_demo",
        },
        search={"provider": "fixture"},
        fetch={"offline": True},
    )
    setup_logging()
    raw = settings.path("data/raw")
    if not raw.exists() or not any(raw.iterdir()):
        ensure_demo_corpus(settings.root)
    kb = KnowledgeBase(settings)
    if kb.stats()["chunks"] == 0:
        kb.ingest([raw], reset=True)
    args.topic = args.topic or "中国新能源汽车行业（2024）"
    return _run_pipeline(args, settings=settings)


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _build_settings(args)
    setup_logging()
    qa_path = settings.path(args.qa_file)
    if not qa_path.exists():
        print(f"评测集不存在：{qa_path}")
        return 1
    qa = load_qa(qa_path)
    if args.limit:
        qa = qa[: args.limit]
    kb = KnowledgeBase(settings)
    if kb.stats()["chunks"] == 0:
        print("知识库为空，请先运行：python -m app.cli bootstrap")
        return 1

    retrieval = evaluate_retrieval(
        kb,
        qa,
        top_k_values=args.top_k_values,
        thresholds=args.thresholds,
    )
    print("\n检索评测（命中率 / MRR）")
    print(
        f"{'top_k':>5} {'threshold':>9} {'hit_rate':>9} {'MRR':>7} "
        f"{'avg_ms':>8} {'跳过不可答':>10}  缺失"
    )
    for row in retrieval:
        print(
            f"{row['top_k']:>5} {row['threshold']:>9.2f} {row['hit_rate']:>9.2%} "
            f"{row['mrr']:>7.3f} {row['avg_latency_ms']:>8.1f} "
            f"{row['skipped_unanswerable']:>10}  {','.join(row['missed'])}"
        )

    payload: dict[str, Any] = {
        "retrieval": retrieval,
        "qa_set": str(qa_path),
        "kb": kb.stats(),
    }
    if args.with_qa:
        if settings.llm.provider == "mock":
            print("\n--with-qa 需要真实模型 provider，Mock 模式仅输出检索指标。")
        else:
            print("\n闭卷 vs RAG 问答评测（接入真实模型）...")
            llm = create_llm(settings.llm)
            qa_result = evaluate_qa(
                llm,
                kb,
                qa,
                top_k=args.top_k_values[0],
                threshold=args.thresholds[0],
            )
            payload["qa"] = qa_result
            closed_ci = qa_result["closed_book_ci95"]
            rag_ci = qa_result["rag_ci95"]
            print(
                f"总体准确率：闭卷 {qa_result['closed_book_accuracy']:.2%} "
                f"[95% CI {closed_ci[0]:.2%}-{closed_ci[1]:.2%}] → "
                f"RAG {qa_result['rag_accuracy']:.2%} "
                f"[95% CI {rag_ci[0]:.2%}-{rag_ci[1]:.2%}] "
                f"({qa_result['delta_pp']:+.1f}pp)"
            )
            print(
                f"可回答题：闭卷 {qa_result['closed_book_answerable_accuracy']:.2%} → "
                f"RAG {qa_result['rag_answerable_accuracy']:.2%}；"
                f"不可回答题 RAG 拒答率 {qa_result['rag_refusal_accuracy']:.2%}"
            )
            print("分类准确率：")
            for category, values in qa_result["by_category"].items():
                print(
                    f"  - {category}: n={values['n']}, "
                    f"闭卷={values['closed_accuracy']:.2%}, RAG={values['rag_accuracy']:.2%}"
                )

    out_dir = settings.path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n评测结果已写入 {out_path}")
    return 0


def cmd_build_eval(args: argparse.Namespace) -> int:
    from app.pipeline.qa_builder import build_eval_set

    settings = _build_settings(args)
    setup_logging()
    kb = KnowledgeBase(settings)
    if kb.stats()["chunks"] == 0:
        print("知识库为空，请先运行：python -m app.cli bootstrap")
        return 1
    provider = settings.llm.provider
    llm = None
    if provider != "mock":
        llm = create_llm(settings.llm)
        print(f"使用真实模型构建评测集：{provider}/{settings.llm.model}")
    else:
        print("使用离线规则抽取构建评测集（无需 API Key）")

    result = build_eval_set(
        kb,
        llm=llm,
        per_chunk=args.per_chunk,
        max_chunks=args.max_chunks,
        sample=args.sample,
        seed_path=None if args.no_seed else settings.path("data/eval/qa_seed.jsonl"),
        legacy_path=None if args.no_legacy else settings.path("data/eval/qa.jsonl"),
        include_unanswerable=not args.no_unanswerable,
        out_path=settings.path(args.out),
    )
    stats = result["stats"]
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(
        "\n提示：自动生成/规则抽取的题目必须人工抽检 evidence_quote；"
        "正式写简历建议扩到 60-100 题，并加入私有研报与不可回答题。"
    )
    if stats["total"] < 50:
        print(
            "当前题量偏少。可增加研报数量、提高 --per-chunk，"
            "或使用 --provider deepseek 让模型生成更丰富的题目。"
        )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("未安装 streamlit，请执行 pip install -r requirements.txt")
        return 1
    target = ROOT / "app" / "ui" / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(target),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true" if args.headless else "false",
    ]
    return subprocess.call(command, cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insight", description="IndustryInsight 多 Agent 行业调研系统"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="生成示例研报并构建本地知识库")
    boot.set_defaults(func=cmd_bootstrap)

    ingest = sub.add_parser("ingest", help="解析 PDF/Word/Markdown 并写入向量库")
    ingest.add_argument("paths", nargs="*", help="文件或目录，默认 data/raw")
    ingest.add_argument("--reset", action="store_true", help="清空索引后重建")
    ingest.set_defaults(func=cmd_ingest)

    run = sub.add_parser("run", help="运行完整研究流水线")
    run.add_argument("--topic", required=True, help="研究主题")
    run.add_argument("--focus", default="", help="用户关注点")
    run.add_argument("--provider", default=None, help="mock/openai/deepseek/qwen/moonshot/zhipu")
    run.add_argument(
        "--thinking",
        choices=["disabled", "low", "high", "max"],
        default=None,
        help="深度思考强度；智谱 glm-5.x 不支持 disabled",
    )
    run.add_argument("--model", default=None, help="覆盖模型名")
    run.add_argument("--online", action="store_true", help="启用真实联网检索与网页抓取")
    run.add_argument("--top-k", type=int, default=None, help="RAG 检索 Top-K")
    run.add_argument("--threshold", type=float, default=None, help="RAG 相似度阈值")
    run.add_argument("--target-chars", type=int, default=None, help="报告目标字数")
    run.add_argument("--output-dir", default=None, help="运行产物目录")
    run.set_defaults(func=_run_pipeline)

    demo = sub.add_parser("demo", help="离线一键演示（Mock + fixtures）")
    demo.add_argument("--topic", default=None)
    demo.add_argument("--target-chars", type=int, default=8000)
    demo.set_defaults(func=cmd_demo)

    evaluation = sub.add_parser("eval", help="RAG 检索/问答评测")
    evaluation.add_argument(
        "--qa-file", default="data/eval/qa.jsonl", help="评测集 JSONL 路径"
    )
    evaluation.add_argument("--top-k", dest="top_k_values", nargs="+", type=int, default=[2, 4, 8])
    evaluation.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.0, 0.18, 0.3]
    )
    evaluation.add_argument("--with-qa", action="store_true", help="使用真实 LLM 评测闭卷/RAG 准确率")
    evaluation.add_argument("--provider", default=None)
    evaluation.add_argument(
        "--thinking",
        choices=["disabled", "low", "high", "max"],
        default=None,
        help="深度思考强度；智谱 glm-5.x 不支持 disabled",
    )
    evaluation.add_argument("--model", default=None)
    evaluation.add_argument("--limit", type=int, default=None)
    evaluation.set_defaults(func=cmd_eval)

    build_eval = sub.add_parser("build-eval", help="从知识库构建评测集（规则抽取或 LLM 生成）")
    build_eval.add_argument("--provider", default=None)
    build_eval.add_argument(
        "--thinking",
        choices=["disabled", "low", "high", "max"],
        default=None,
        help="深度思考强度；智谱 glm-5.x 不支持 disabled",
    )
    build_eval.add_argument("--model", default=None)
    build_eval.add_argument("--per-chunk", type=int, default=3, help="每个切片最多生成多少题")
    build_eval.add_argument("--max-chunks", type=int, default=30)
    build_eval.add_argument(
        "--sample",
        choices=["head", "random", "stratified"],
        default="head",
        help="切片采样方式：head 取前 N 个、random 随机、stratified 按来源分层",
    )
    build_eval.add_argument("--out", default="data/eval/qa_v2.jsonl")
    build_eval.add_argument("--no-legacy", action="store_true", help="不合并已有 qa.jsonl")
    build_eval.add_argument("--no-seed", action="store_true", help="不合并人工种子题")
    build_eval.add_argument("--no-unanswerable", action="store_true", help="不加入不可回答题")
    build_eval.set_defaults(func=cmd_build_eval)

    ui = sub.add_parser("ui", help="启动 Streamlit 前端")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--headless", action="store_true")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"配置错误：{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
