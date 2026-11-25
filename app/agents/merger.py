"""证据合并：确定性节点，不调用 LLM，统一编号后供分析与写作引用。"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from app.config import TaskSpec
from app.core.agent import Agent
from app.core.types import TaskResult
from app.schemas import EvidenceBundle


class EvidenceUnavailableError(RuntimeError):
    """所有检索分支都没有产出可引用证据。"""


class MergerAgent(Agent):
    def run(self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.perf_counter()
        bundles = []
        for result in deps.values():
            if result.kind not in {"web_research", "kb_research"}:
                continue
            try:
                bundles.append(EvidenceBundle.model_validate(result.parsed or {}))
            except Exception:  # noqa: BLE001
                continue
        if bundles:
            merged = bundles[0].merge(*bundles[1:])
        else:
            merged = EvidenceBundle(topic=str(inputs.get("topic", "")))
        merged.topic = str(inputs.get("topic") or merged.topic)
        merged.reindex()
        if not merged.items:
            raise EvidenceUnavailableError(
                "未获得任何可用证据：联网检索与本地知识库均未返回可引用条目。"
                "请检查联网检索配置（如 TAVILY_API_KEY）或向知识库导入相关研报后重试。"
            )
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "llm_latency_ms": 0.0,
        }
        return TaskResult(
            task_id=task.id,
            agent=self.spec.key,
            kind=task.kind,
            title=task.title or task.id,
            text=json.dumps(merged.model_dump(), ensure_ascii=False, indent=2),
            parsed=merged.model_dump(),
            usage=usage,
            started_at=started_at,
            ended_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
