"""配置加载：settings / agents / tasks 三层配置，支持环境变量与运行时覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from jinja2 import Template

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """配置缺失或非法。"""


PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "mock": {"model": "mock-researcher", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "openai": {"model": "gpt-4o-mini", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "deepseek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "moonshot": {
        "model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "zhipu": {
        "model": "glm-4.6",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "api_key_env": "ZHIPU_API_KEY",
        # 智谱默认开启深度思考，会让短答案变成 reasoning 并耗尽 max_tokens
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}


@dataclass
class LLMSettings:
    provider: str = "mock"
    model: str = ""
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    extra_body: Optional[dict[str, Any]] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 2
    json_mode: bool = False

    def __post_init__(self) -> None:
        preset = PROVIDER_PRESETS.get(self.provider)
        if preset is None:
            raise ConfigError(
                f"未知 LLM provider: {self.provider}，可选 {sorted(PROVIDER_PRESETS)}"
            )
        if not self.model:
            self.model = preset["model"]
        if not self.base_url:
            self.base_url = preset["base_url"]
        if not self.api_key_env:
            self.api_key_env = preset["api_key_env"]
        if self.extra_body is None:
            self.extra_body = preset.get("extra_body")
        # 智谱 glm-5.x 始终思考，不支持 disabled；自动降级为低强度思考
        if self.provider == "zhipu" and self.model.startswith("glm-5"):
            if not self.extra_body or self.extra_body.get("thinking", {}).get("type") == "disabled":
                self.extra_body = {
                    "thinking": {"type": "enabled", "effort": "low"},
                    "reasoning_effort": "low",
                }


@dataclass
class RAGSettings:
    backend: str = "local"
    embedding: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key_env: Optional[str] = None
    embedding_base_url: Optional[str] = None
    dim: int = 768
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    score_threshold: float = 0.18
    persist_dir: str = "data/index"
    chroma_dir: str = "data/chroma"
    collection: str = "industry_reports"


@dataclass
class SearchSettings:
    provider: str = "fixture"
    max_results: int = 5
    timeout: int = 20
    fixture_dir: str = "data/fixtures/search"
    tavily_api_key_env: str = "TAVILY_API_KEY"


@dataclass
class FetchSettings:
    offline: bool = True
    timeout: int = 20
    max_chars: int = 6000
    fixture_dir: str = "data/fixtures/pages"


@dataclass
class PipelineSettings:
    max_react_steps: int = 7
    max_coverage_rounds: int = 2
    max_validation_retries: int = 2
    max_task_retries: int = 1
    target_chars: int = 8000
    parallel: bool = True
    max_workers: int = 2
    output_dir: str = "runs"
    max_dep_chars: int = 14000
    max_observation_chars: int = 3000


@dataclass
class ObservabilitySettings:
    input_price_per_1k: float = 0.00015
    output_price_per_1k: float = 0.0006


@dataclass
class Settings:
    root: Path
    llm: LLMSettings = field(default_factory=LLMSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    fetch: FetchSettings = field(default_factory=FetchSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)

    @classmethod
    def from_file(cls, path: Path | str | None = None, root: Path | None = None) -> "Settings":
        path = Path(path) if path else ROOT / "config" / "settings.yaml"
        if not path.exists():
            raise ConfigError(f"配置文件不存在: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        llm_raw = dict(raw.get("llm", {}))
        for env_key, field_name in (
            ("INSIGHT_PROVIDER", "provider"),
            ("INSIGHT_MODEL", "model"),
            ("INSIGHT_BASE_URL", "base_url"),
            ("INSIGHT_API_KEY_ENV", "api_key_env"),
        ):
            value = os.getenv(env_key)
            if value:
                llm_raw[field_name] = value
        root = Path(root) if root else path.resolve().parent.parent
        return cls(
            root=root,
            llm=LLMSettings(**llm_raw),
            rag=RAGSettings(**raw.get("rag", {})),
            search=SearchSettings(**raw.get("search", {})),
            fetch=FetchSettings(**raw.get("fetch", {})),
            pipeline=PipelineSettings(**raw.get("pipeline", {})),
            observability=ObservabilitySettings(**raw.get("observability", {})),
        )

    def with_overrides(
        self,
        *,
        llm: dict[str, Any] | None = None,
        rag: dict[str, Any] | None = None,
        search: dict[str, Any] | None = None,
        fetch: dict[str, Any] | None = None,
        pipeline: dict[str, Any] | None = None,
    ) -> "Settings":
        llm_raw = dict(llm or {})
        # 切换 provider 或模型时清空上一家的 base_url / api_key_env / extra_body
        if "provider" in llm_raw and llm_raw["provider"] != self.llm.provider:
            llm_raw.setdefault("base_url", None)
            llm_raw.setdefault("api_key_env", None)
            llm_raw.setdefault("extra_body", None)
        if "model" in llm_raw and llm_raw["model"] != self.llm.model:
            llm_raw.setdefault("extra_body", None)
        return replace(
            self,
            llm=replace(self.llm, **llm_raw),
            rag=replace(self.rag, **(rag or {})),
            search=replace(self.search, **(search or {})),
            fetch=replace(self.fetch, **(fetch or {})),
            pipeline=replace(self.pipeline, **(pipeline or {})),
        )

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)

    def api_key(self) -> Optional[str]:
        if not self.llm.api_key_env:
            return None
        key = os.getenv(self.llm.api_key_env, "")
        return key.strip() or None

    def require_api_key(self) -> str:
        key = self.api_key()
        if not key:
            raise ConfigError(
                f"未找到 {self.llm.api_key_env}。请在 .env 中配置，"
                f"或使用 provider=mock 的离线演示模式。"
            )
        return key


@dataclass
class AgentSpec:
    key: str
    role: str
    goal: str
    backstory: str = ""
    tools: list[str] = field(default_factory=list)
    mode: str = "direct"
    max_steps: int = 7
    temperature: Optional[float] = None


@dataclass
class TaskSpec:
    id: str
    agent: str
    kind: str
    description: str
    title: str = ""
    depends_on: list[str] = field(default_factory=list)
    tools: Optional[list[str]] = None
    mode: Optional[str] = None
    output_schema: Optional[str] = None
    coverage_check: bool = False
    min_chars: int = 0
    min_citations: int = 0
    require_table: bool = False
    required_sections: list[str] = field(default_factory=list)

    def render_description(self, inputs: dict[str, Any]) -> str:
        safe_inputs = dict(inputs)
        safe_inputs.setdefault("focus", "")
        return Template(self.description).render(**safe_inputs).strip()


def load_agents(path: Path | str | None = None) -> dict[str, AgentSpec]:
    path = Path(path) if path else ROOT / "config" / "agents.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed = {
        "role",
        "goal",
        "backstory",
        "tools",
        "mode",
        "max_steps",
        "temperature",
    }
    specs: dict[str, AgentSpec] = {}
    for key, item in raw.items():
        payload = {k: v for k, v in (item or {}).items() if k in allowed}
        if "role" not in payload or "goal" not in payload:
            raise ConfigError(f"agents.yaml 中 {key} 缺少 role/goal")
        specs[key] = AgentSpec(key=key, **payload)
    return specs


def load_tasks(path: Path | str | None = None) -> list[TaskSpec]:
    path = Path(path) if path else ROOT / "config" / "tasks.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks: list[TaskSpec] = []
    for item in raw.get("tasks", []):
        tasks.append(
            TaskSpec(
                id=item["id"],
                agent=item["agent"],
                kind=item["kind"],
                description=item.get("description", ""),
                title=item.get("title", item["id"]),
                depends_on=list(item.get("depends_on", [])),
                tools=item.get("tools"),
                mode=item.get("mode"),
                output_schema=item.get("output_schema"),
                coverage_check=bool(item.get("coverage_check", False)),
                min_chars=int(item.get("min_chars", 0)),
                min_citations=int(item.get("min_citations", 0)),
                require_table=bool(item.get("require_table", False)),
                required_sections=list(item.get("required_sections", [])),
            )
        )
    if not tasks:
        raise ConfigError("tasks.yaml 中没有任务定义")
    return tasks
