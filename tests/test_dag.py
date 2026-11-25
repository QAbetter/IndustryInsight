from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path

from app.config import TaskSpec
from app.core.crew import Crew, CrewConfigError
from app.core.tracker import RunTracker
from app.core.types import TaskResult


class DummyAgent:
    def __init__(self, name: str, *, barrier: threading.Barrier | None = None, fail: bool = False):
        self.name = name
        self.barrier = barrier
        self.fail = fail
        self.seen_deps: dict[str, set[str]] = {}

    def run(self, task, deps, inputs):
        self.seen_deps[task.id] = set(deps)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        text = f"{task.id}:{'+'.join(sorted(deps)) or 'root'}"
        return TaskResult(
            task_id=task.id,
            agent=self.name,
            kind=task.kind,
            title=task.title,
            text=text,
            parsed={"deps": sorted(deps)},
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            started_at=datetime.now().isoformat(),
            ended_at=datetime.now().isoformat(),
            duration_ms=1.0,
        )


def make_task(task_id: str, deps: list[str]) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        agent="researcher",
        kind="dummy",
        description="",
        title=task_id,
        depends_on=deps,
    )


class CrewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tracker = RunTracker("test", Path(self.tmp.name))

    def tearDown(self):
        self.tracker.close()
        self.tmp.cleanup()

    def test_dag_order_and_parallel_branches(self):
        tasks = [
            make_task("plan", []),
            make_task("web", ["plan"]),
            make_task("kb", ["plan"]),
            make_task("merge", ["web", "kb"]),
        ]
        barrier = threading.Barrier(2)
        agents = {
            "plan": DummyAgent("researcher"),
            "web": DummyAgent("researcher", barrier=barrier),
            "kb": DummyAgent("researcher", barrier=barrier),
            "merge": DummyAgent("researcher"),
        }
        crew = Crew(tasks, agents, self.tracker, parallel=True, max_workers=2, max_task_retries=0)
        result = crew.run({"topic": "x"})
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(agents["web"].seen_deps["web"], {"plan"})
        self.assertEqual(agents["merge"].seen_deps["merge"], {"web", "kb"})
        merged_text = result.outputs["merge"].text
        self.assertTrue(merged_text.startswith("merge:"))
        self.assertIn("web", merged_text)
        self.assertIn("kb", merged_text)

    def test_cycle_detection(self):
        tasks = [make_task("a", ["b"]), make_task("b", ["a"])]
        agents = {"a": DummyAgent("r"), "b": DummyAgent("r")}
        with self.assertRaises(CrewConfigError):
            Crew(tasks, agents, self.tracker)

    def test_failure_blocks_dependents(self):
        tasks = [make_task("a", []), make_task("b", ["a"]), make_task("c", [])]
        agents = {
            "a": DummyAgent("r", fail=True),
            "b": DummyAgent("r"),
            "c": DummyAgent("r"),
        }
        crew = Crew(tasks, agents, self.tracker, parallel=False, max_task_retries=0)
        result = crew.run({"topic": "x"})
        self.assertIn("a", result.errors)
        self.assertIn("b", result.errors)
        self.assertIn("c", result.outputs)
        self.assertNotIn("b", result.outputs)


if __name__ == "__main__":
    unittest.main()
