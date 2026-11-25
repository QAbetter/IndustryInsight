"""撰写员 Agent：按大纲分章节长文写作，最后统一渲染引用、表格与局限说明。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any

from app.agents.prompts import build_writer_system_prompt
from app.config import TaskSpec
from app.core.agent import Agent, AgentOutputError
from app.core.structured import check_markdown, count_citations
from app.core.types import TaskResult
from app.schemas import EvidenceBundle

logger = logging.getLogger("insight.writer")

DEFAULT_SECTIONS = [
    "核心摘要",
    "行业概览与市场规模",
    "竞争格局",
    "产业链与商业模式",
    "趋势与风险",
    "结论与建议",
]
_EXCLUDED = ("参考", "附录", "数据来源说明")


def _has_table(text: str) -> bool:
    return bool(re.search(r"^\s*\|.+\|\s*$", text, re.M)) and bool(
        re.search(r"^\s*\|[\s\-:|]+\|\s*$", text, re.M)
    )


class WriterAgent(Agent):
    def run(self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.perf_counter()
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "llm_latency_ms": 0.0,
        }
        self._tool_calls = []
        self._validation_retries = 0

        topic = str(inputs.get("topic") or "行业研究")
        plan = self._dep_parsed(deps, "research_plan") or {}
        analysis = self._dep_parsed(deps, "analysis") or {"topic": topic}
        evidence = self._collect_evidence(deps)

        sections = [
            str(item)
            for item in (plan.get("report_outline") or DEFAULT_SECTIONS)
            if not any(flag in str(item) for flag in _EXCLUDED)
        ][:8]
        if len(sections) < 4:
            sections = DEFAULT_SECTIONS

        target_chars = int(inputs.get("target_chars") or self.settings.pipeline.target_chars)
        per_section = max(420, target_chars // max(1, len(sections)))
        contents: list[str] = []
        for index, title in enumerate(sections, start=1):
            content = self._write_section(
                title=title,
                index=index,
                total=len(sections),
                outline=sections,
                topic=topic,
                analysis=analysis,
                evidence=evidence,
                per_section=per_section,
            )
            contents.append(content)

        body = "\n\n".join(contents)
        if not _has_table(body):
            table = self._metrics_table(analysis)
            if table:
                body += "\n\n" + table

        limitations = self._limitations(analysis, evidence)
        references = self._references(evidence)
        body = "\n\n".join(part for part in (body, limitations, references) if part)

        header = (
            f"# {topic}——行业研究报告\n\n"
            f"> 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')} ｜ "
            f"证据 {len(evidence.items)} 条 ｜ 正文约 {len(body)} 字\n"
            "> 本报告由 IndustryInsight 多 Agent 流水线生成；数据来自本地研报与联网检索结果，"
            "不构成投资建议。"
        )
        report = f"{header}\n\n{body}"

        contract = self._contract(task)
        errors = check_markdown(
            report,
            min_chars=contract.min_chars,
            required_sections=contract.required_sections,
            min_citations=min(contract.min_citations, len(evidence.items)),
            require_table=contract.require_table,
        )
        if errors:
            raise AgentOutputError(f"[{task.id}] 报告交付校验失败：{'；'.join(errors)}")

        parsed = {
            "topic": topic,
            "sections": sections,
            "chars": len(report),
            "citations": count_citations(report),
            "evidence_count": len(evidence.items),
        }
        ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return TaskResult(
            task_id=task.id,
            agent=self.spec.key,
            kind=task.kind,
            title=task.title or task.id,
            text=report,
            parsed=parsed,
            usage=self._usage.copy(),
            tool_calls=list(self._tool_calls),
            validation_retries=self._validation_retries,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    # ------------------------------------------------------------------ #
    def _write_section(
        self,
        *,
        title: str,
        index: int,
        total: int,
        outline: list[str],
        topic: str,
        analysis: dict[str, Any],
        evidence: EvidenceBundle,
        per_section: int,
    ) -> str:
        system = build_writer_system_prompt(self.spec)
        outline_text = "\n".join(f"{i}. {name}" for i, name in enumerate(outline, start=1))
        user = (
            f"## 报告主题\n{topic}\n\n"
            f"## 全文章节\n{outline_text}\n\n"
            f"## 当前章节（第 {index}/{total} 节）\n"
            f"<SECTION_TITLE>{title}</SECTION_TITLE>\n\n"
            "## 结构化分析（唯一事实来源）\n<ANALYSIS_JSON>\n"
            f"{json.dumps(analysis, ensure_ascii=False, indent=2)}\n"
            "</ANALYSIS_JSON>\n\n"
            "## 可用证据（引用时必须使用这些编号）\n<EVIDENCE_JSON>\n"
            f"{json.dumps(evidence.compact(max_items=30, detail_chars=320), ensure_ascii=False)}\n"
            "</EVIDENCE_JSON>\n\n"
            "## 写作要求\n"
            f"- 本节约 {per_section} 字，从二级标题“## {title}”开始；\n"
            "- 关键数字后用 [E#] 标注证据，编号不得超出证据清单；\n"
            "- 适合对比的内容使用 Markdown 表格；\n"
            "- 不要重复其他章节，不要输出参考资料（系统统一生成）；\n"
            "- 不编造证据编号、公司名和数字，缺失信息写“公开资料未披露”。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        min_chars = max(150, int(per_section * 0.3))
        content = ""
        for attempt in range(self.settings.pipeline.max_validation_retries + 1):
            response = self._chat(messages, stage=f"report_writing#{index}")
            content = self._clean_section(response.text, title)
            if len(content) >= min_chars:
                return content
            self._validation_retries += 1
            self.tracker.validation_failed(
                agent=self.spec.key,
                stage=f"report_writing#{index}",
                attempt=attempt + 1,
                errors=[f"章节 {title} 仅 {len(content)} 字，低于 {min_chars} 字"],
            )
            if attempt >= self.settings.pipeline.max_validation_retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": (
                            f"本节字数不足（{len(content)} 字，至少 {min_chars} 字）。请扩写并保留证据引用，"
                            "补充机制解释、数据对比或边界条件，不要重复前文。只输出完整的本节内容。"
                        ),
                    },
                ]
            )
        if len(content) < 120:
            raise AgentOutputError(f"章节 {title} 生成内容过短（{len(content)} 字）")
        return content

    @staticmethod
    def _clean_section(text: str, title: str) -> str:
        content = text.strip()
        content = re.sub(r"^```(?:markdown|md)?\s*", "", content, flags=re.I)
        content = re.sub(r"\s*```$", "", content).strip()
        if content.startswith("# ") and not content.startswith("## "):
            content = "#" + content
        if not content.startswith("#"):
            content = f"## {title}\n\n{content}"
        return content

    @staticmethod
    def _dep_parsed(deps: dict[str, TaskResult], kind: str) -> dict[str, Any] | None:
        for result in deps.values():
            if result.kind == kind and isinstance(result.parsed, dict):
                return result.parsed
        return None

    @staticmethod
    def _collect_evidence(deps: dict[str, TaskResult]) -> EvidenceBundle:
        for result in deps.values():
            if result.kind == "merge_evidence" and isinstance(result.parsed, dict):
                try:
                    return EvidenceBundle.model_validate(result.parsed)
                except Exception:  # noqa: BLE001
                    break
        bundles: list[EvidenceBundle] = []
        for result in deps.values():
            if result.kind in {"web_research", "kb_research"} and isinstance(result.parsed, dict):
                try:
                    bundles.append(EvidenceBundle.model_validate(result.parsed))
                except Exception:  # noqa: BLE001
                    continue
        if not bundles:
            return EvidenceBundle()
        merged = bundles[0]
        for bundle in bundles[1:]:
            merged = merged.merge(bundle)
        return merged

    @staticmethod
    def _metrics_table(analysis: dict[str, Any]) -> str:
        rows = analysis.get("key_metrics") or []
        if rows:
            lines = [
                "## 关键指标一览",
                "",
                "| 指标 | 数值 | 年份 | 证据 |",
                "| --- | --- | --- | --- |",
            ]
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                source_ids = "、".join(f"[{item}]" for item in row.get("source_ids", [])) or "—"
                lines.append(
                    f"| {row.get('name', '')} | {row.get('value', '')} | "
                    f"{row.get('year', '') or '—'} | {source_ids} |"
                )
            return "\n".join(lines)
        competitors = analysis.get("competitive_landscape") or []
        if competitors:
            lines = [
                "## 竞争格局一览",
                "",
                "| 企业 | 份额 | 定位 | 证据 |",
                "| --- | --- | --- | --- |",
            ]
            for row in competitors[:10]:
                if not isinstance(row, dict):
                    continue
                source_ids = "、".join(f"[{item}]" for item in row.get("source_ids", [])) or "—"
                lines.append(
                    f"| {row.get('name', '')} | {row.get('share', '')} | "
                    f"{row.get('positioning', '')} | {source_ids} |"
                )
            return "\n".join(lines)
        return ""

    @staticmethod
    def _limitations(analysis: dict[str, Any], evidence: EvidenceBundle) -> str:
        items: list[str] = []
        for gap in (analysis.get("gaps") or []) + (evidence.gaps or []):
            text = str(gap).strip()
            if text and text not in items:
                items.append(text)
        for contradiction in analysis.get("contradictions") or []:
            text = str(contradiction).strip()
            if text:
                items.append(f"数据冲突：{text}")
        if not items:
            items = ["部分细分市场缺少统一口径的公开数据，结论需结合后续数据更新。"]
        return "## 研究局限与数据缺口\n\n" + "\n".join(f"- {item}" for item in items[:8])

    @staticmethod
    def _references(evidence: EvidenceBundle) -> str:
        if not evidence.items:
            return "## 参考资料\n\n本次运行未获得可引用的外部证据。"
        lines = ["## 参考资料", ""]
        for item in evidence.items[:30]:
            if item.url:
                suffix = f"（{item.published_at}）" if item.published_at else ""
                lines.append(f"- [{item.id}] {item.title or item.claim}. {item.url} {suffix}")
            else:
                page = f"第 {item.page} 页" if item.page else "本地研报"
                lines.append(f"- [{item.id}] {item.source or item.title}（{page}）")
        lines.append("")
        lines.append("> 证据编号与正文引用一一对应；演示模式数据仅用于工程验证。")
        return "\n".join(lines)
