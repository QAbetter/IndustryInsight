"""研究员 Agent：ReAct 检索 + 证据充分性评估 + 触发二次补搜。"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import Any

from app.agents.prompts import COVERAGE, build_system_prompt, kind_instruction
from app.config import TaskSpec
from app.core.agent import Agent, AgentOutputError
from app.core.types import TaskResult
from app.schemas import CoverageDecision, EvidenceBundle

logger = logging.getLogger("insight.researcher")


class ResearcherAgent(Agent):
    def run(self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult:
        started = time.perf_counter()
        result = super().run(task, deps, inputs)
        if not task.coverage_check:
            return result

        evidence = EvidenceBundle.model_validate(result.parsed or {})
        rounds = 0
        while rounds < self.settings.pipeline.max_coverage_rounds:
            round_no = rounds + 1
            try:
                decision = self._assess_coverage(task, evidence, deps, inputs, round_no)
            except Exception as exc:  # noqa: BLE001 - 质检失败不应中断主流程
                logger.warning("证据充分性评估失败，保留现有证据：%s", exc)
                break
            self.tracker.coverage(
                agent=self.spec.key, round_no=round_no, decision=decision.model_dump()
            )
            if decision.sufficient:
                # 质检通过时以最新判断为准，清除已被补搜解决的历史缺口
                evidence.gaps = list(decision.gaps)
                break
            for gap in decision.gaps:
                if gap and gap not in evidence.gaps:
                    evidence.gaps.append(gap)
            if not decision.followup_queries:
                break
            logger.info(
                "证据不足，触发第 %d 轮补搜：%s", round_no, decision.followup_queries
            )
            messages = self._followup_messages(task, evidence, decision, inputs, round_no)
            try:
                _, parsed = self.run_react(
                    task,
                    deps,
                    inputs,
                    messages=messages,
                    stage=f"{task.kind}#followup{round_no}",
                )
            except AgentOutputError as exc:
                logger.warning("补充检索未产出合规证据，保留现有结果：%s", exc)
                break
            new_bundle = EvidenceBundle.model_validate(parsed or {})
            evidence = evidence.merge(new_bundle)
            rounds += 1

        result.usage = self._usage.copy()
        result.tool_calls = list(self._tool_calls)
        result.validation_retries = self._validation_retries
        result.text = json.dumps(evidence.model_dump(), ensure_ascii=False, indent=2)
        result.parsed = evidence.model_dump()
        result.duration_ms = (time.perf_counter() - started) * 1000
        return replace(result)

    def _assess_coverage(
        self,
        task: TaskSpec,
        evidence: EvidenceBundle,
        deps: dict[str, TaskResult],
        inputs: dict[str, Any],
        round_no: int,
    ) -> CoverageDecision:
        plan_text = ""
        for dep in deps.values():
            if dep.kind == "research_plan":
                plan_text = dep.text
                break
        payload = {
            "round": round_no,
            "topic": inputs.get("topic", ""),
            "evidence": evidence.compact(max_items=24, detail_chars=320),
            "gaps": evidence.gaps,
            "queries": evidence.queries,
        }
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.spec, tools_text="", instruction=COVERAGE, react=False
                ),
            },
            {
                "role": "user",
                "content": (
                    "## 研究计划\n"
                    f"{plan_text}\n\n"
                    "## 当前证据包\n<EVIDENCE_JSON>\n"
                    f"{json.dumps(payload, ensure_ascii=False)}\n"
                    "</EVIDENCE_JSON>\n\n"
                    "请判断证据是否足以支撑计划中的全部子问题，并给出补充检索建议。"
                ),
            },
        ]
        model = self._structured_call(
            messages, CoverageDecision, stage="coverage_check", max_retries=1
        )
        return CoverageDecision.model_validate(model.model_dump())

    def _followup_messages(
        self,
        task: TaskSpec,
        evidence: EvidenceBundle,
        decision: CoverageDecision,
        inputs: dict[str, Any],
        round_no: int,
    ) -> list[dict[str, str]]:
        queries = "；".join(decision.followup_queries[:3])
        allowed = self._allowed_tools(task)
        system = build_system_prompt(
            self.spec,
            tools_text=self.registry.describe(allowed),
            instruction=kind_instruction(task.kind),
            react=True,
        )
        user = (
            f"## 补充检索轮次 {round_no}\n"
            f"研究主题：{inputs.get('topic', '')}\n\n"
            "## 质检发现的信息缺口\n"
            + "\n".join(f"- {gap}" for gap in decision.gaps or ["证据覆盖不完整"])
            + f"\n\n补充检索查询：{queries}\n\n"
            "## 已有证据（不要重复）\n<EVIDENCE_JSON>\n"
            f"{json.dumps(evidence.compact(max_items=20, detail_chars=240), ensure_ascii=False)}\n"
            "</EVIDENCE_JSON>\n\n"
            "请调用工具执行上述补充检索，并输出本轮**新增**证据（最多 6 条，"
            "detail 不超过 120 字）；不要重复已有证据，系统会自动合并。"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
