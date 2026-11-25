"""任务执行的统一交付结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskResult:
    task_id: str
    agent: str
    kind: str
    title: str
    text: str
    parsed: Any = None
    usage: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    validation_retries: int = 0
    started_at: str = ""
    ended_at: str = ""
    duration_ms: float = 0.0
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "kind": self.kind,
            "title": self.title,
            "parsed": self.parsed,
            "usage": self.usage,
            "tool_calls": self.tool_calls,
            "validation_retries": self.validation_retries,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class CrewResult:
    outputs: dict[str, TaskResult]
    errors: dict[str, str]
    started_at: str
    ended_at: str
    duration_ms: float

    @property
    def ok(self) -> bool:
        return not self.errors
