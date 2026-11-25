"""端到端流水线：配置 -> 知识库 -> 工具 -> Agent -> Task DAG -> 产物落盘。"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from app.agents.factory import build_agents
from app.config import Settings, load_agents, load_tasks
from app.core.crew import Crew
from app.core.logging_utils import setup_logging, shutdown_logging
from app.core.tools import build_tool_registry
from app.core.tracker import RunTracker
from app.rag.knowledge import KnowledgeBase
from app.reporting.renderer import render_report_html

logger = logging.getLogger("insight.pipeline")


class PipelineError(RuntimeError):
    def __init__(self, message: str, run_dir: Optional[Path] = None):
        super().__init__(message)
        self.run_dir = run_dir


@dataclass
class PipelineResult:
    run_id: str
    run_dir: Path
    topic: str
    report: str
    metrics: dict[str, Any]
    errors: dict[str, str]
    artifacts: dict[str, str]
    plan: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.report)


def _slug(text: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text).strip("-")
    return (cleaned or fallback)[:24]


class ResearchPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ):
        self.settings = settings
        self.on_event = on_event
        self.agent_specs = load_agents(settings.root / "config" / "agents.yaml")
        self.tasks = load_tasks(settings.root / "config" / "tasks.yaml")

    def run(
        self, topic: str, focus: str = "", target_chars: int | None = None
    ) -> PipelineResult:
        topic = topic.strip()
        if not topic:
            raise PipelineError("研究主题不能为空")
        target_chars = int(target_chars or self.settings.pipeline.target_chars)
        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(topic)}"
        run_dir = self.settings.path(self.settings.pipeline.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(run_dir / "run.log")
        tracker = RunTracker(
            run_id,
            run_dir,
            on_event=self.on_event,
            input_price_per_1k=self.settings.observability.input_price_per_1k,
            output_price_per_1k=self.settings.observability.output_price_per_1k,
        )
        started = time.perf_counter()
        tracker.emit(
            "run_start",
            topic=topic,
            provider=self.settings.llm.provider,
            model=self.settings.llm.model,
            search=self.settings.search.provider,
            rag={"backend": self.settings.rag.backend, "top_k": self.settings.rag.top_k, "threshold": self.settings.rag.score_threshold},
            target_chars=target_chars,
        )

        try:
            kb = self._prepare_knowledge_base(tracker)
            registry = build_tool_registry(self.settings, kb)
            agents = build_agents(
                self.tasks, self.agent_specs, self.settings, registry, tracker
            )
            crew = Crew(
                self.tasks,
                agents,
                tracker,
                parallel=self.settings.pipeline.parallel,
                max_workers=self.settings.pipeline.max_workers,
                max_task_retries=self.settings.pipeline.max_task_retries,
            )
            inputs = {
                "topic": topic,
                "focus": focus.strip(),
                "target_chars": target_chars,
                "today": datetime.now().strftime("%Y-%m-%d"),
            }
            crew_result = crew.run(inputs)
            artifacts = self._save_artifacts(run_dir, crew_result.outputs, tracker)

            merge_error = crew_result.errors.get("merge_evidence")
            if merge_error:
                raise PipelineError(f"证据阶段失败：{merge_error}", run_dir)

            report_result = crew_result.outputs.get("write")
            if report_result is None:
                raise PipelineError(
                    f"报告任务未完成：{crew_result.errors.get('write', '未知原因')}", run_dir
                )
            report = report_result.text
            plan = self._parsed(crew_result.outputs.get("plan"))
            analysis = self._parsed(crew_result.outputs.get("analyze"))
            evidence = self._parsed(crew_result.outputs.get("merge_evidence"))

            meta = {
                "topic": topic,
                "run_id": run_id,
                "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s": round((time.perf_counter() - started), 1),
                "chars": len(report),
                "evidence_count": len((evidence or {}).get("items", [])),
                "citations": len(set(re.findall(r"\[E\d+\]", report))),
                "provider": self.settings.llm.provider,
                "model": self.settings.llm.model,
                "top_k": self.settings.rag.top_k,
                "threshold": self.settings.rag.score_threshold,
            }
            html_path = run_dir / "report.html"
            html_path.write_text(
                render_report_html(report, meta, tracker.summary(), evidence or {}),
                encoding="utf-8",
            )
            artifacts["report_html"] = str(html_path)

            status = "ok" if not crew_result.errors else "partial"
            tracker.run_end(status)
            metrics = tracker.save()
            run_info = {
                "run_id": run_id,
                "topic": topic,
                "status": status,
                "errors": crew_result.errors,
                "metrics": metrics,
                "artifacts": artifacts,
                "meta": meta,
            }
            (run_dir / "run.json").write_text(
                json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            tracker.emit("run_summary", status=status, elapsed_ms=round(elapsed_ms, 1))
            tracker.close()
            shutdown_logging()
            return PipelineResult(
                run_id=run_id,
                run_dir=run_dir,
                topic=topic,
                report=report,
                metrics=metrics,
                errors=crew_result.errors,
                artifacts=artifacts,
                plan=plan,
                evidence=evidence,
                analysis=analysis,
                elapsed_ms=elapsed_ms,
                status=status,
            )
        except Exception as exc:
            logger.exception("流水线执行失败")
            tracker.emit("run_error", error=str(exc))
            tracker.run_end("error")
            try:
                tracker.save()
            finally:
                tracker.close()
                shutdown_logging()
            if isinstance(exc, PipelineError):
                raise
            raise PipelineError(str(exc), run_dir) from exc

    # ------------------------------------------------------------------ #
    def _prepare_knowledge_base(self, tracker: RunTracker) -> KnowledgeBase:
        kb = KnowledgeBase(self.settings)
        stats = kb.stats()
        if stats["chunks"] == 0:
            raw = self.settings.path("data/raw")
            raw.mkdir(parents=True, exist_ok=True)
            if not any(raw.iterdir()):
                from app.sample_data import ensure_demo_corpus

                logger.info("未找到本地研报，自动生成演示语料")
                ensure_demo_corpus(self.settings.root)
            logger.info("向量库为空，自动构建索引：%s", raw)
            ingest_stats = kb.ingest([raw], reset=True)
            tracker.emit("kb_ingest", **ingest_stats.to_dict())
            stats = kb.stats()
        tracker.emit("kb_ready", **stats)
        if stats["chunks"] == 0:
            logger.warning("知识库为空，kb_search 将返回空结果")
        return kb

    @staticmethod
    def _parsed(result) -> dict[str, Any] | None:
        if result is None:
            return None
        return result.parsed if isinstance(result.parsed, dict) else None

    def _save_artifacts(self, run_dir: Path, outputs: dict, tracker: RunTracker) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        mapping = {
            "plan": "plan.json",
            "collect_web": "evidence_web.json",
            "collect_kb": "evidence_kb.json",
            "merge_evidence": "evidence.json",
            "analyze": "analysis.json",
        }
        for task_id, filename in mapping.items():
            result = outputs.get(task_id)
            if result is None:
                continue
            path = run_dir / filename
            if isinstance(result.parsed, dict):
                path.write_text(
                    json.dumps(result.parsed, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                path.write_text(result.text, encoding="utf-8")
            artifacts[task_id] = str(path)

        report_result = outputs.get("write")
        if report_result:
            report_path = run_dir / "report.md"
            report_path.write_text(report_result.text, encoding="utf-8")
            artifacts["report"] = str(report_path)

        events_path = run_dir / "events.jsonl"
        if events_path.exists():
            artifacts["events"] = str(events_path)
        for name, path in artifacts.items():
            tracker.emit("artifact", name=name, path=path)
        return artifacts
