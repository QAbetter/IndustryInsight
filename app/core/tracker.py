"""可观测性：逐事件记录 Token / 耗时 / 工具调用 / 校验重试，并聚合瓶颈。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("insight.tracker")


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunTracker:
    def __init__(
        self,
        run_id: str,
        run_dir: Path | str,
        *,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
        input_price_per_1k: float = 0.0,
        output_price_per_1k: float = 0.0,
    ):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.input_price_per_1k = input_price_per_1k
        self.output_price_per_1k = output_price_per_1k
        self._on_event = on_event
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._file = self.events_path.open("a", encoding="utf-8")
        self.started_at = _now_iso()
        self._perf_start = time.perf_counter()
        self.ended_at: Optional[str] = None
        self.status = "running"

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {"ts": _now_iso(), "type": event_type, "run_id": self.run_id, **payload}
        with self._lock:
            self._events.append(event)
            self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._file.flush()
        if self._on_event:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001 - 回调失败不影响主流程
                logger.debug("事件回调异常", exc_info=True)
        return event

    def task_start(self, task_id: str, agent: str, kind: str, title: str = "") -> None:
        self.emit("task_start", task_id=task_id, agent=agent, kind=kind, title=title or task_id)

    def task_end(
        self,
        task_id: str,
        agent: str,
        kind: str,
        *,
        duration_ms: float,
        status: str = "ok",
        output_chars: int = 0,
        error: str = "",
    ) -> None:
        self.emit(
            "task_end",
            task_id=task_id,
            agent=agent,
            kind=kind,
            duration_ms=round(duration_ms, 1),
            status=status,
            output_chars=output_chars,
            error=error,
        )

    def llm_call(
        self,
        *,
        agent: str,
        stage: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        truncated: bool = False,
    ) -> None:
        self.emit(
            "llm_call",
            agent=agent,
            stage=stage,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 1),
            truncated=truncated,
        )

    def tool_call(
        self,
        *,
        agent: str,
        stage: str,
        tool: str,
        args_preview: str,
        ok: bool,
        duration_ms: float,
        output_chars: int = 0,
        error: str = "",
    ) -> None:
        self.emit(
            "tool_call",
            agent=agent,
            stage=stage,
            tool=tool,
            args=args_preview[:300],
            ok=ok,
            duration_ms=round(duration_ms, 1),
            output_chars=output_chars,
            error=error[:300],
        )

    def validation_failed(self, *, agent: str, stage: str, attempt: int, errors: list[str]) -> None:
        self.emit(
            "validation_failed",
            agent=agent,
            stage=stage,
            attempt=attempt,
            errors=errors[:8],
        )

    def coverage(self, *, agent: str, round_no: int, decision: dict[str, Any]) -> None:
        self.emit("coverage_check", agent=agent, round=round_no, **decision)

    def run_end(self, status: str = "ok") -> None:
        self.status = status
        self.ended_at = _now_iso()
        self.emit("run_end", status=status, duration_ms=round(self.elapsed_ms(), 1))

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._perf_start) * 1000

    def summary(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)

        by_agent: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "llm_calls": 0,
                "llm_latency_ms": 0.0,
                "tool_calls": 0,
                "tool_failures": 0,
                "tool_latency_ms": 0.0,
                "task_duration_ms": 0.0,
                "validation_failures": 0,
            }
        )
        by_stage: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "llm_calls": 0,
                "latency_ms": 0.0,
                "tool_calls": 0,
                "tool_failures": 0,
            }
        )
        prompt_tokens = completion_tokens = llm_calls = tool_calls = tool_failures = 0
        validation_failures = 0
        coverage_checks = 0
        for event in events:
            kind = event.get("type")
            if kind == "llm_call":
                prompt = int(event.get("prompt_tokens", 0))
                completion = int(event.get("completion_tokens", 0))
                latency = float(event.get("latency_ms", 0))
                agent = event.get("agent", "unknown")
                stage = event.get("stage", "unknown")
                prompt_tokens += prompt
                completion_tokens += completion
                llm_calls += 1
                by_agent[agent]["prompt_tokens"] += prompt
                by_agent[agent]["completion_tokens"] += completion
                by_agent[agent]["llm_calls"] += 1
                by_agent[agent]["llm_latency_ms"] += latency
                by_stage[stage]["prompt_tokens"] += prompt
                by_stage[stage]["completion_tokens"] += completion
                by_stage[stage]["llm_calls"] += 1
                by_stage[stage]["latency_ms"] += latency
            elif kind == "tool_call":
                tool_calls += 1
                agent = event.get("agent", "unknown")
                stage = event.get("stage", "unknown")
                by_agent[agent]["tool_calls"] += 1
                by_agent[agent]["tool_latency_ms"] += float(event.get("duration_ms", 0))
                by_stage[stage]["tool_calls"] += 1
                if not event.get("ok", False):
                    tool_failures += 1
                    by_agent[agent]["tool_failures"] += 1
                    by_stage[stage]["tool_failures"] += 1
            elif kind == "validation_failed":
                validation_failures += 1
                by_agent[event.get("agent", "unknown")]["validation_failures"] += 1
            elif kind == "coverage_check":
                coverage_checks += 1
            elif kind == "task_end":
                agent = event.get("agent", "unknown")
                by_agent[agent]["task_duration_ms"] += float(event.get("duration_ms", 0))

        cost = (
            prompt_tokens / 1000 * self.input_price_per_1k
            + completion_tokens / 1000 * self.output_price_per_1k
        )
        for values in by_agent.values():
            for key in ("llm_latency_ms", "tool_latency_ms", "task_duration_ms"):
                values[key] = round(values[key], 1)
        for values in by_stage.values():
            values["latency_ms"] = round(values["latency_ms"], 1)

        stage_ranking = sorted(
            (
                {
                    "stage": stage,
                    "latency_ms": values["latency_ms"],
                    "llm_calls": values["llm_calls"],
                    "tokens": values["prompt_tokens"] + values["completion_tokens"],
                }
                for stage, values in by_stage.items()
            ),
            key=lambda item: (item["latency_ms"], item["tokens"]),
            reverse=True,
        )
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": round(self.elapsed_ms(), 1),
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "cost_usd": round(cost, 6),
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "tool_failures": tool_failures,
            "validation_failures": validation_failures,
            "coverage_checks": coverage_checks,
            "by_agent": dict(by_agent),
            "by_stage": dict(by_stage),
            "slowest_stages": stage_ranking[:3],
            "bottlenecks": stage_ranking[:3],
        }

    def save(self) -> dict[str, Any]:
        summary = self.summary()
        (self.run_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:  # noqa: BLE001
                pass
