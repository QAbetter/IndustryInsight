"""统一 Agent 执行层：Direct 与 ReAct 两种运行模式、工具权限、格式校验与重试。"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

from app.agents.prompts import build_system_prompt, kind_instruction
from app.config import AgentSpec, Settings, TaskSpec
from app.core.llm import BaseLLM
from app.core.structured import (
    check_markdown,
    correction_prompt,
    count_citations,
    extract_json,
    validate_json_output,
)
from app.core.tools import ToolRegistry
from app.core.tracker import RunTracker
from app.core.types import TaskResult
from app.schemas import SCHEMA_MAP

logger = logging.getLogger("insight.agent")

_ACTION_RE = re.compile(r"^\s*Action\s*[:：]\s*([A-Za-z_][\w\-]*)\s*$", re.M)
_ACTION_INPUT_RE = re.compile(
    r"Action\s*Input\s*[:：]\s*(.*?)(?=\n\s*(?:Thought|Observation|Action)\s*[:：]|\Z)",
    re.S | re.I,
)
_FINAL_RE = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S | re.I)


class AgentError(RuntimeError):
    pass


class AgentOutputError(AgentError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _truncation_hint(contract: "OutputContract") -> str:
    if contract.schema is not None:
        return (
            "上一条输出因长度上限被截断。请重新输出**完整合法**的 JSON，并压缩内容："
            "证据最多 8 条，每条 detail 不超过 120 字，claim 一句话；"
            "不要输出解释、示例或多余字段。信息装不下时写入 gaps。"
        )
    return "上一条输出因长度上限被截断。请压缩篇幅，直接输出完整的最终交付物。"


def parse_action(text: str) -> Optional[tuple[str, Any]]:
    match = _ACTION_RE.search(text)
    if not match:
        return None
    name = match.group(1).strip()
    input_match = _ACTION_INPUT_RE.search(text)
    raw = input_match.group(1).strip() if input_match else ""
    if not raw:
        return name, {}
    raw = re.sub(r"^```(?:json|JSON)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()
    if raw.startswith("{"):
        try:
            return name, extract_json(raw)
        except ValueError:
            return name, raw
    return name, raw


def extract_final_answer(text: str) -> Optional[str]:
    match = _FINAL_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


@dataclass
class OutputContract:
    schema: Optional[type[BaseModel]] = None
    min_chars: int = 0
    required_sections: list[str] = field(default_factory=list)
    min_citations: int = 0
    require_table: bool = False


class Agent:
    def __init__(
        self,
        spec: AgentSpec,
        llm: BaseLLM,
        registry: ToolRegistry,
        tracker: RunTracker,
        settings: Settings,
    ):
        self.spec = spec
        self.llm = llm
        self.registry = registry
        self.tracker = tracker
        self.settings = settings
        self._usage: dict[str, Any] = {}
        self._tool_calls: list[dict[str, Any]] = []
        self._validation_retries = 0

    # ------------------------------------------------------------------ #
    # 公共入口
    # ------------------------------------------------------------------ #
    def run(self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult:
        started_at = _now()
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

        mode = task.mode or self.spec.mode
        if mode == "long_form":
            text, parsed = self.run_long_form(task, deps, inputs)
        elif mode == "react":
            text, parsed = self.run_react(task, deps, inputs)
        else:
            text, parsed = self.run_direct(task, deps, inputs)

        ended_at = _now()
        return TaskResult(
            task_id=task.id,
            agent=self.spec.key,
            kind=task.kind,
            title=task.title or task.id,
            text=text,
            parsed=parsed,
            usage=self._usage.copy(),
            tool_calls=list(self._tool_calls),
            validation_retries=self._validation_retries,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def run_long_form(
        self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]
    ) -> tuple[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__} 不支持 long_form 模式")

    # ------------------------------------------------------------------ #
    # 运行模式
    # ------------------------------------------------------------------ #
    def run_direct(
        self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]
    ) -> tuple[str, Any]:
        messages = self._base_messages(task, deps, inputs, react=False)
        contract = self._contract(task)
        max_retries = self.settings.pipeline.max_validation_retries
        last_errors: list[str] = []
        for attempt in range(max_retries + 1):
            response = self._chat(messages, stage=task.kind)
            if response.finish_reason == "length":
                parsed, errors = None, ["输出因 max_tokens 上限被截断"]
            else:
                parsed, errors = self._validate_output(response.text, contract, task)
            if not errors:
                return response.text.strip(), parsed
            last_errors = errors
            self._on_invalid(task, attempt, errors)
            if attempt >= max_retries:
                break
            hint = kind_instruction(task.kind)
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": correction_prompt(errors, hint)},
                ]
            )
        raise AgentOutputError(
            f"[{task.id}] 结构化输出校验失败：{'；'.join(last_errors[:5])}"
        )

    def run_react(
        self,
        task: TaskSpec,
        deps: dict[str, TaskResult],
        inputs: dict[str, Any],
        *,
        messages: Optional[list[dict[str, str]]] = None,
        stage: Optional[str] = None,
    ) -> tuple[str, Any]:
        messages = messages or self._base_messages(task, deps, inputs, react=True)
        stage = stage or task.kind
        contract = self._contract(task)
        limits = [
            value
            for value in (self.spec.max_steps, self.settings.pipeline.max_react_steps)
            if value and value > 0
        ]
        max_steps = min(limits) if limits else 7
        max_retries = self.settings.pipeline.max_validation_retries
        final_attempts = 0
        last_errors: list[str] = []

        for _step in range(max_steps):
            response = self._chat(messages, stage=stage)
            text = response.text.strip()
            if response.finish_reason == "length":
                self._on_invalid(task, final_attempts, ["输出因 max_tokens 上限被截断"])
                final_attempts += 1
                if final_attempts > max_retries:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": _truncation_hint(contract)},
                    ]
                )
                continue
            if _FINAL_RE.search(text):
                final_text = extract_final_answer(text) or text
                parsed, errors = self._validate_output(final_text, contract, task)
                if not errors:
                    return final_text.strip(), parsed
                last_errors = errors
                final_attempts += 1
                self._on_invalid(task, final_attempts - 1, errors)
                if final_attempts > max_retries:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": correction_prompt(errors, kind_instruction(task.kind)),
                        },
                    ]
                )
                continue

            action = parse_action(text)
            if action:
                name, raw_args = action
                allowed = self._allowed_tools(task)
                result = self.registry.execute(name, raw_args, allowed)
                self._tool_calls.append(
                    {
                        "step": len(self._tool_calls) + 1,
                        "tool": name,
                        "ok": result.ok,
                        "duration_ms": round(result.duration_ms, 1),
                        "args": result.args,
                    }
                )
                self.tracker.tool_call(
                    agent=self.spec.key,
                    stage=stage,
                    tool=name,
                    args_preview=json.dumps(result.args, ensure_ascii=False, default=str),
                    ok=result.ok,
                    duration_ms=result.duration_ms,
                    output_chars=len(result.output),
                    error=result.error,
                )
                observation = result.observation()
                limit = self.settings.pipeline.max_observation_chars
                if limit and len(observation) > limit:
                    observation = observation[:limit] + "\n...[Observation 过长，已截断]"
                messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": f"Observation: {observation}"},
                    ]
                )
                continue

            # 未使用标准标记但直接给出了交付物，按最终答案处理
            parsed, errors = self._validate_output(text, contract, task)
            if not errors:
                return text, parsed
            last_errors = errors
            final_attempts += 1
            self._on_invalid(task, final_attempts - 1, errors)
            if final_attempts > max_retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": correction_prompt(errors, kind_instruction(task.kind)),
                    },
                ]
            )

        # 收尾：强制基于已有 Observation 交付
        messages.append(
            {
                "role": "user",
                "content": (
                    "已达到工具调用/校验上限。请立即基于已有 Observation 输出 Final Answer，"
                    "严格遵守交付契约；无法获取的数据写“未获取”。"
                    + (
                        "证据最多 8 条，detail 不超过 120 字，避免输出被截断。"
                        if contract.schema is not None
                        else ""
                    )
                ),
            }
        )
        for attempt in range(max_retries + 1):
            response = self._chat(messages, stage=f"{stage}#finalize")
            final_text = extract_final_answer(response.text) or response.text.strip()
            if response.finish_reason == "length":
                last_errors = ["输出因 max_tokens 上限被截断"]
            else:
                parsed, errors = self._validate_output(final_text, contract, task)
                if not errors:
                    return final_text.strip(), parsed
                last_errors = errors
            self._on_invalid(task, attempt, last_errors)
            if attempt >= max_retries:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": _truncation_hint(contract)
                        if response.finish_reason == "length"
                        else correction_prompt(last_errors, kind_instruction(task.kind)),
                    },
                ]
            )
        raise AgentOutputError(
            f"[{task.id}] ReAct 流程未产出合规交付物：{'；'.join(last_errors[:5])}"
        )

    # ------------------------------------------------------------------ #
    # 消息构建与校验
    # ------------------------------------------------------------------ #
    def _base_messages(
        self,
        task: TaskSpec,
        deps: dict[str, TaskResult],
        inputs: dict[str, Any],
        *,
        react: bool,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt(task, react=react)},
            {"role": "user", "content": self._user_prompt(task, deps, inputs)},
        ]

    def _system_prompt(self, task: TaskSpec, *, react: bool) -> str:
        allowed = self._allowed_tools(task)
        tools_text = self.registry.describe(allowed) if react else ""
        return build_system_prompt(
            self.spec,
            tools_text=tools_text,
            instruction=kind_instruction(task.kind),
            react=react,
        )

    def _user_prompt(
        self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]
    ) -> str:
        sections = [
            f"## 当前任务：{task.title or task.id}（{task.id}）",
            task.render_description(inputs),
            "## 全局输入",
            f"- 研究主题：{inputs.get('topic', '')}",
            f"- 用户关注点：{inputs.get('focus') or '无'}",
            f"- 报告目标字数：{inputs.get('target_chars', self.settings.pipeline.target_chars)}",
            f"- 当前日期：{inputs.get('today', '')}",
        ]
        if deps:
            sections.append("## 依赖任务交付物")
            for task_id, result in deps.items():
                sections.append(f"### {task_id}（{result.kind}）")
                payload = result.text.strip()
                limit = self.settings.pipeline.max_dep_chars
                if len(payload) > limit:
                    payload = payload[:limit] + "\n...[依赖交付物过长，已截断]"
                sections.append(payload)
        sections.append(
            "## 执行提醒\n"
            "严格按系统提示中的工作协议执行；最终交付物必须满足交付契约，"
            "不要输出与任务无关的寒暄。"
        )
        return "\n\n".join(sections)

    def _allowed_tools(self, task: TaskSpec) -> set[str]:
        allowed = set(self.spec.tools)
        if task.tools is not None:
            allowed &= set(task.tools)
        return allowed

    def _contract(self, task: TaskSpec) -> OutputContract:
        schema = SCHEMA_MAP.get(task.output_schema) if task.output_schema else None
        return OutputContract(
            schema=schema,
            min_chars=task.min_chars,
            required_sections=list(task.required_sections),
            min_citations=task.min_citations,
            require_table=task.require_table,
        )

    def _validate_output(
        self, text: str, contract: OutputContract, task: TaskSpec
    ) -> tuple[Any, list[str]]:
        if contract.schema is not None:
            model, errors = validate_json_output(text, contract.schema)
            return (model.model_dump() if model else None), errors
        errors = check_markdown(
            text,
            min_chars=contract.min_chars,
            required_sections=contract.required_sections,
            min_citations=contract.min_citations,
            require_table=contract.require_table,
        )
        return {"chars": len(text), "citations": count_citations(text)}, errors

    def _on_invalid(self, task: TaskSpec, attempt: int, errors: list[str]) -> None:
        self._validation_retries += 1
        self.tracker.validation_failed(
            agent=self.spec.key, stage=task.kind, attempt=attempt + 1, errors=errors
        )
        logger.warning("[%s] 第 %d 次输出校验失败：%s", task.id, attempt + 1, errors[:3])

    # ------------------------------------------------------------------ #
    # LLM 调用与统计
    # ------------------------------------------------------------------ #
    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: str,
        temperature: Optional[float] = None,
    ) -> Any:
        response = self.llm.chat(
            messages,
            stage=stage,
            temperature=temperature if temperature is not None else self.spec.temperature,
        )
        self._usage["prompt_tokens"] += response.prompt_tokens
        self._usage["completion_tokens"] += response.completion_tokens
        self._usage["total_tokens"] += response.total_tokens
        self._usage["llm_calls"] += 1
        self._usage["llm_latency_ms"] += response.latency_ms
        self.tracker.llm_call(
            agent=self.spec.key,
            stage=stage,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
            truncated=response.finish_reason == "length",
        )
        return response

    def _structured_call(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        *,
        stage: str,
        max_retries: Optional[int] = None,
    ) -> BaseModel:
        retries = (
            self.settings.pipeline.max_validation_retries if max_retries is None else max_retries
        )
        last_errors: list[str] = []
        for attempt in range(retries + 1):
            response = self._chat(messages, stage=stage)
            model, errors = validate_json_output(response.text, schema)
            if model is not None:
                return model
            last_errors = errors
            self._validation_retries += 1
            self.tracker.validation_failed(
                agent=self.spec.key, stage=stage, attempt=attempt + 1, errors=errors
            )
            if attempt >= retries:
                break
            messages = messages + [
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": correction_prompt(errors)},
            ]
        raise AgentOutputError(f"[{stage}] 结构化输出校验失败：{'；'.join(last_errors[:5])}")
