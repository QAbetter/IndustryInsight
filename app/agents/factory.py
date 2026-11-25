"""根据 YAML 配置构建每个 Task 对应的 Agent 实例（每个任务独立实例，便于并行）。"""

from __future__ import annotations

from app.agents.analyst import AnalystAgent
from app.agents.merger import MergerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.writer import WriterAgent
from app.config import AgentSpec, Settings, TaskSpec
from app.core.agent import Agent
from app.core.llm import BaseLLM, create_llm
from app.core.tools import ToolRegistry
from app.core.tracker import RunTracker


def _agent_class(task: TaskSpec, spec: AgentSpec):
    if task.kind == "merge_evidence":
        return MergerAgent
    if task.kind == "report_writing" or spec.mode == "long_form":
        return WriterAgent
    if spec.key == "researcher":
        return ResearcherAgent
    if spec.key == "analyst":
        return AnalystAgent
    return Agent


def build_agents(
    tasks: list[TaskSpec],
    specs: dict[str, AgentSpec],
    settings: Settings,
    registry: ToolRegistry,
    tracker: RunTracker,
    llm: BaseLLM | None = None,
) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    for task in tasks:
        spec = specs.get(task.agent)
        if spec is None:
            raise KeyError(f"任务 {task.id} 引用了未定义角色 {task.agent}")
        task_llm = llm or create_llm(settings.llm)
        agent_cls = _agent_class(task, spec)
        agents[task.id] = agent_cls(spec, task_llm, registry, tracker, settings)
    return agents
