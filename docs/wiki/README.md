# IndustryInsight Code Wiki

> 面向开发者的代码级项目文档。目标：读完之后能够独立跑起来、定位任意功能的实现位置、并在不破坏既有契约的前提下扩展新角色 / 新工具 / 新数据源。

## 1. 项目是什么

IndustryInsight 是一个 **多 Agent 行业调研与数据分析系统**，把以下能力串成一条可观测、可评测的流水线：

```
研究规划 → 联网检索 ∥ 私有研报检索 → 证据合并 → 结构化分析 → 分章节长文撰写 → Markdown / HTML 报告
```

- 三个核心角色：**研究员（researcher）**、**分析师（analyst）**、**撰写员（writer）**，另有确定性的 **证据合并器（merger）**。
- 两种运行环境：`mock` 离线演示（无需任何 API Key）与真实 OpenAI 兼容模型（OpenAI / DeepSeek / Qwen / Moonshot / 智谱）。
- 两条证据来源：`search_web` / `fetch_page` 联网检索，`kb_search` 本地 RAG 知识库（PDF / DOCX / MD / TXT）。
- 全链路可观测：`events.jsonl`（逐事件）+ `metrics.json`（聚合指标）+ `run.json`（运行元信息）。

## 2. 文档地图

| 文档 | 内容 | 适合谁 |
| --- | --- | --- |
| [01-architecture.md](./01-architecture.md) | 整体架构、分层、运行时序、Task DAG、目录结构、关键设计决策 | 所有人，先读 |
| [02-core-runtime.md](./02-core-runtime.md) | 编排内核：Agent 运行时、Crew 调度、LLM 抽象、结构化校验、工具层、检索层、可观测性 | 后端 / 平台开发 |
| [03-agents.md](./03-agents.md) | 四个 Agent 的职责与算法、Prompt 体系、Pydantic 交付契约 | Agent / Prompt 开发 |
| [04-rag.md](./04-rag.md) | 文档解析、递归切片、Embedding、向量库、混合检索 | RAG / 检索调优 |
| [05-pipeline-and-eval.md](./05-pipeline-and-eval.md) | 端到端编排、产物落盘、检索/问答评测、评测集构建、HTML 渲染 | 评测 / 数据工程 |
| [06-config-and-interfaces.md](./06-config-and-interfaces.md) | 配置系统、CLI、Streamlit、示例语料生成 | 使用与集成方 |
| [07-dependencies.md](./07-dependencies.md) | 模块依赖图、分层约束、第三方依赖、可选依赖与降级策略 | 架构 / 维护者 |
| [08-operations.md](./08-operations.md) | 环境搭建、运行命令、配置配方、测试、扩展指南、常见故障 | 运维 / 贡献者 |

## 3. 项目速览

| 维度 | 说明 |
| --- | --- |
| 语言 / 运行时 | Python 3.10+（代码使用 `X | None`、`from __future__ import annotations`） |
| Web 框架 | 无传统后端框架；前端为 **Streamlit** 单页应用，报告渲染使用 **Jinja2** 模板 |
| Agent 编排 | 自研轻量内核（参考 CrewAI 的 Agent / Task / Crew 抽象），支持 DAG 并行与失败隔离 |
| LLM 接入 | OpenAI 兼容 SDK；`mock` 为离线脚本化实现 |
| 向量库 | 内置 NumPy 持久化实现（默认）/ Chroma（可选） |
| 数据校验 | Pydantic v2（证据类模型 `extra="forbid"`） |
| 测试 | `unittest`，8 个测试文件、36 个测试函数，覆盖协议解析、DAG、工具、RAG、评测、端到端 Mock |
| 入口 | `python -m app.cli <command>`；Streamlit 入口 `app/ui/streamlit_app.py` |

主要入口文件：

- [app/cli.py](../../app/cli.py) — 命令行总入口（bootstrap / ingest / run / demo / build-eval / eval / ui）
- [app/pipeline/runner.py](../../app/pipeline/runner.py) — 端到端编排
- [app/ui/streamlit_app.py](../../app/ui/streamlit_app.py) — Web UI
- [config/settings.yaml](../../config/settings.yaml) — 运行参数
- [config/agents.yaml](../../config/agents.yaml) — 角色与工具权限
- [config/tasks.yaml](../../config/tasks.yaml) — 任务、依赖与交付契约

## 4. 五个必须先理解的概念

1. **AgentSpec / TaskSpec**：角色与任务都是 YAML 配置，运行时加载为 dataclass。Agent 的可用工具 = `角色声明工具 ∩ 任务声明工具`。
2. **OutputContract（交付契约）**：每个任务声明 `output_schema`（Pydantic 模型）或 Markdown 约束（最少字数、必需章节、最少引用、必需表格）。不达标 → 纠错重试 → 仍失败则任务失败。
3. **EvidenceBundle（证据包）**：系统中唯一的事实载体。证据在合并阶段被统一重编号为 `E1..En`，分析师与撰写员都只能引用这套编号。
4. **Crew（Task DAG）**：按依赖就绪执行，`collect_web` 与 `collect_kb` 并行；单任务失败会被隔离，下游任务跳过，无关分支继续。
5. **RunTracker（可观测性）**：所有 LLM 调用、工具调用、校验失败、补搜轮次都写入 `events.jsonl`，运行结束聚合为 `metrics.json`。

## 5. 五分钟跑通（离线）

```powershell
# 1) 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) 生成演示研报 + 构建本地知识库
python -m app.cli bootstrap

# 3) 一键离线演示（Mock LLM + fixtures 检索）
python -m app.cli demo

# 4) 打开前端
python -m app.cli ui
```

产物在 `runs/<run_id>/`：`report.md`、`report.html`、`plan.json`、`evidence.json`、`analysis.json`、`metrics.json`、`events.jsonl`、`run.json`、`run.log`。

真实模型只需在 `.env` 配置 Key 后追加 `--provider deepseek --online`，详见 [06-config-and-interfaces.md](./06-config-and-interfaces.md)。

## 6. 推荐阅读路径

- **想改 Agent 行为 / Prompt**：01 → 03 → 02（Agent 运行时与校验）
- **想调 RAG 效果**：04 → 05（评测）→ 08（调参流程）
- **想接新模型 / 新搜索源**：06（配置与 Provider）→ 02（LLM / Search 抽象）→ 07（依赖）
- **想加一个新角色或新任务**：03（factory 与 YAML）→ 06（tasks.yaml）→ 08（扩展指南）
- **想部署 / 排障**：08 → 05（产物与指标）

## 7. 文档约定

- 路径链接均相对于本目录；形如 `../../app/core/agent.py` 指向仓库源码。
- 代码块中若出现参数名，以 `config/settings.yaml` 的实际值为准；dataclass 默认值可能与之不同。
- “证据”指 `EvidenceItem`，“切片”指 RAG 的 `Chunk`，两者不是同一概念。
- 文档描述的是当前仓库状态；若源码变更，请优先信任源码。
