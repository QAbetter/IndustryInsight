"""Task 依赖图调度：拓扑就绪 + 分支并行 + 失败隔离。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

from app.config import TaskSpec
from app.core.agent import Agent
from app.core.tracker import RunTracker
from app.core.types import CrewResult, TaskResult

logger = logging.getLogger("insight.crew")


class CrewConfigError(RuntimeError):
    pass


class Crew:
    def __init__(
        self,
        tasks: list[TaskSpec],
        agents: dict[str, Agent],
        tracker: RunTracker,
        *,
        parallel: bool = True,
        max_workers: int = 2,
        max_task_retries: int = 1,
    ):
        self.task_list = tasks
        self.tasks = {task.id: task for task in tasks}
        self.agents = agents
        self.tracker = tracker
        self.parallel = parallel
        self.max_workers = max(1, max_workers)
        self.max_task_retries = max(0, max_task_retries)
        self._validate()

    def _validate(self) -> None:
        if len(self.tasks) != len(self.task_list):
            raise CrewConfigError("存在重复的 task id")
        for task in self.tasks.values():
            if task.id not in self.agents:
                raise CrewConfigError(f"任务 {task.id} 没有绑定 Agent")
            for dependency in task.depends_on:
                if dependency not in self.tasks:
                    raise CrewConfigError(
                        f"任务 {task.id} 依赖了不存在的任务 {dependency}"
                    )
                if dependency == task.id:
                    raise CrewConfigError(f"任务 {task.id} 不能依赖自身")
        self._detect_cycle()

    def _detect_cycle(self) -> None:
        colors: dict[str, int] = {}

        def visit(task_id: str, trail: list[str]) -> None:
            color = colors.get(task_id, 0)
            if color == 1:
                cycle = " -> ".join(trail + [task_id])
                raise CrewConfigError(f"Task 依赖图存在环：{cycle}")
            if color == 2:
                return
            colors[task_id] = 1
            for dependency in self.tasks[task_id].depends_on:
                visit(dependency, trail + [task_id])
            colors[task_id] = 2

        for task_id in self.tasks:
            visit(task_id, [])

    def run(self, inputs: dict[str, Any]) -> CrewResult:
        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        started = time.perf_counter()
        results: dict[str, TaskResult] = {}
        errors: dict[str, str] = {}
        pending: dict[str, TaskSpec] = dict(self.tasks)

        while pending:
            blocked = [
                task
                for task in pending.values()
                if any(dependency in errors for dependency in task.depends_on)
            ]
            if blocked:
                for task in blocked:
                    failed = [dep for dep in task.depends_on if dep in errors]
                    errors[task.id] = f"依赖任务失败，已跳过：{failed}"
                    self.tracker.emit(
                        "task_skipped", task_id=task.id, agent=task.agent, reason=errors[task.id]
                    )
                    del pending[task.id]
                continue

            ready = [
                task
                for task in pending.values()
                if all(dependency in results for dependency in task.depends_on)
            ]
            if not ready:
                for task in pending.values():
                    errors[task.id] = "依赖图无法满足"
                break

            if self.parallel and len(ready) > 1:
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                    futures = {
                        pool.submit(
                            self._run_task,
                            task,
                            {dep: results[dep] for dep in task.depends_on},
                            inputs,
                        ): task
                        for task in ready
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            results[task.id] = future.result()
                        except Exception as exc:  # noqa: BLE001 - 单任务失败不拖垮整体
                            errors[task.id] = str(exc)
                            logger.error("任务 %s 失败：%s", task.id, exc)
            else:
                for task in ready:
                    try:
                        results[task.id] = self._run_task(
                            task, {dep: results[dep] for dep in task.depends_on}, inputs
                        )
                    except Exception as exc:  # noqa: BLE001
                        errors[task.id] = str(exc)
                        logger.error("任务 %s 失败：%s", task.id, exc)

            for task in ready:
                pending.pop(task.id, None)

        duration_ms = (time.perf_counter() - started) * 1000
        ended_at = datetime.now().astimezone().isoformat(timespec="seconds")
        return CrewResult(
            outputs=results,
            errors=errors,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
        )

    def _run_task(
        self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]
    ) -> TaskResult:
        agent = self.agents[task.id]
        self.tracker.task_start(task.id, task.agent, task.kind, task.title)
        started = time.perf_counter()
        last_error: Optional[Exception] = None
        for attempt in range(self.max_task_retries + 1):
            try:
                result = agent.run(task, deps, inputs)
                self.tracker.task_end(
                    task.id,
                    task.agent,
                    task.kind,
                    duration_ms=result.duration_ms,
                    status="ok",
                    output_chars=len(result.text),
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_task_retries:
                    logger.warning(
                        "任务 %s 第 %d 次执行失败，准备重试：%s", task.id, attempt + 1, exc
                    )
                    self.tracker.emit(
                        "task_retry", task_id=task.id, agent=task.agent, attempt=attempt + 1, error=str(exc)
                    )
                    continue
                break
        duration_ms = (time.perf_counter() - started) * 1000
        self.tracker.task_end(
            task.id,
            task.agent,
            task.kind,
            duration_ms=duration_ms,
            status="error",
            error=str(last_error or ""),
        )
        raise last_error or RuntimeError(f"任务 {task.id} 执行失败")
