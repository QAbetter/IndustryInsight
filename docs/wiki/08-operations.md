# 08 · 运行、测试与运维

## 1. 环境准备

要求：Python 3.10+（已在 3.10.11 / Windows 验证）。

```powershell
# 创建虚拟环境（示例路径按本机 Python 调整）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 基础依赖
pip install -r requirements.txt

# 可选：Chroma 向量库
pip install -r requirements-optional.txt
```

## 2. 最小可运行路径

```powershell
# 1) 生成 3 份演示研报并重建知识库
#    默认 Embedding 为 GLM embedding-3，需要 ZHIPU_API_KEY
python -m app.cli bootstrap

# 2) 离线一键演示（Mock LLM + fixtures 检索 + 独立 hashing 索引）
python -m app.cli demo

# 3) 启动 Streamlit
python -m app.cli ui            # 默认 http://localhost:8501
```

`demo` 的行为：

- 强制 `llm.provider=mock`、`search.provider=fixture`、`fetch.offline=True`；
- 强制 `rag.embedding=hashing`、`persist_dir=data/index_demo`，不需要任何 API Key；
- 若无 `data/raw` 语料则自动生成；
- 知识库为空则自动建索引；
- 默认主题“中国新能源汽车行业（2024）”，默认 8000 字。

> 只想离线跑通可跳过第 1 步直接 `demo`；两者索引目录不同，互不覆盖。

## 3. 真实模型与联网

### 3.1 `.env`

```ini
INSIGHT_PROVIDER=deepseek
INSIGHT_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-xxx

# 默认 GLM embedding-3 向量化需要
ZHIPU_API_KEY=xxx

# 可选：Tavily；不配置时 online 会回退 DuckDuckGo，再回退 fixtures
TAVILY_API_KEY=tvly-xxx
```

### 3.2 运行

```powershell
python -m app.cli run `
  --topic "中国新能源汽车行业（2024）" `
  --focus "重点关注价格战、竞争格局与海外市场" `
  --provider deepseek `
  --online `
  --target-chars 10000
```

### 3.3 其他 Provider

```powershell
# OpenAI
python -m app.cli run --topic "..." --provider openai --model gpt-4o-mini

# Qwen
python -m app.cli run --topic "..." --provider qwen --model qwen-plus

# Moonshot
python -m app.cli run --topic "..." --provider moonshot

# 智谱（glm-4.6 默认关闭深度思考）
python -m app.cli run --topic "..." --provider zhipu --model glm-4.6

# 智谱 glm-5.x 必须开启思考，用低强度
python -m app.cli run --topic "..." --provider zhipu --model glm-5 --thinking low
```

任意 OpenAI 兼容网关：

```ini
INSIGHT_PROVIDER=openai
INSIGHT_BASE_URL=https://your-gateway/v1
INSIGHT_API_KEY_ENV=OPENAI_API_KEY
OPENAI_API_KEY=sk-xxx
```

## 4. 知识库运维

```powershell
# 重建索引（会清空 data/index）
python -m app.cli bootstrap

# 增量导入指定文件或目录
python -m app.cli ingest data/raw
python -m app.cli ingest "D:\reports\ifr_2024.pdf" --reset

# 切换 Chroma
# config/settings.yaml: rag.backend: chroma
pip install -r requirements-optional.txt
python -m app.cli bootstrap
```

数据来源：

- `data/raw/`（支持子目录，如 `data/raw/robotics/`）；
- 入库时按文件名写入 `company` / `report_year` / `report_name` 元数据；
- 从旧版升级后需重建索引，或给已有 `chunks.jsonl` 回填这三个字段，否则 `company` 过滤会返回空结果。
- Streamlit“知识库”页上传（落到 `data/uploads/`）。

## 5. 评测流程

```powershell
# 检索评测（无需 API Key；当前主用机器人行业 501 题题库，取前 100 题对照）
python -m app.cli eval `
  --qa-file data/eval/qa_robotics_v2.jsonl `
  --limit 100 `
  --top-k 8 `
  --thresholds 0.15

# 构建评测集（离线规则抽取）
python -m app.cli build-eval `
  --provider mock --per-chunk 4 --max-chunks 30 --sample stratified `
  --out data/eval/qa_private.jsonl

# 构建评测集（真实模型生成）
python -m app.cli build-eval `
  --provider deepseek --per-chunk 3 `
  --out data/eval/qa_private.jsonl `
  --no-seed --no-legacy

# 闭卷 vs RAG（真实模型）
python -m app.cli eval `
  --qa-file data/eval/qa_robotics_v2.jsonl `
  --limit 100 `
  --provider zhipu --with-qa `
  --top-k 8 --thresholds 0.15
```

结果：`runs/eval_results.json`。

- `retrieval`：每个参数组合的命中率、MRR、平均耗时、未命中题号；
- `qa`（仅 `--with-qa`）：闭卷/RAG 准确率、95% CI、可回答题准确率、拒答率、分类准确率、逐题明细。

## 6. 运行产物与指标

### 6.1 目录

```
runs/
├── <run_id>/
│   ├── report.md
│   ├── report.html
│   ├── plan.json
│   ├── evidence_web.json
│   ├── evidence_kb.json
│   ├── evidence.json
│   ├── analysis.json
│   ├── metrics.json
│   ├── events.jsonl
│   ├── run.json
│   └── run.log
└── eval_results.json
```

### 6.2 `events.jsonl` 事件类型

| type | 触发点 | 主要字段 |
| --- | --- | --- |
| `run_start` | 流水线开始 | topic, provider, model, search, rag, target_chars |
| `kb_ingest` | 自动建库 | files, documents, chunks_added, chunks_skipped, elapsed_ms |
| `kb_ready` | 知识库就绪 | backend, embedder, chunks, sources, load_error |
| `task_start` / `task_end` | 任务边界 | task_id, agent, kind, duration_ms, status, output_chars |
| `task_retry` | 任务级重试 | task_id, attempt, error |
| `task_skipped` | 依赖失败跳过 | task_id, reason |
| `llm_call` | 每次 LLM | agent, stage, model, tokens, latency_ms, truncated |
| `tool_call` | 每次工具 | agent, stage, tool, args, ok, duration_ms, output_chars |
| `validation_failed` | 交付校验失败 | agent, stage, attempt, errors |
| `coverage_check` | 证据质检 | agent, round, sufficient, gaps, followup_queries |
| `artifact` | 产物落盘 | name, path |
| `run_end` / `run_error` | 运行结束 / 异常 | status, duration_ms / error |
| `run_summary` | CLI 汇总 | status, elapsed_ms |

### 6.3 `metrics.json` 关键字段

```json
{
  "run_id": "...", "status": "ok", "started_at": "...", "ended_at": "...",
  "duration_ms": 12345.6,
  "tokens": {"prompt": 0, "completion": 0, "total": 0},
  "cost_usd": 0.0,
  "llm_calls": 0, "tool_calls": 0, "tool_failures": 0,
  "validation_failures": 0, "coverage_checks": 0,
  "by_agent": {"researcher": {"prompt_tokens":0,"completion_tokens":0,"llm_calls":0,
                               "llm_latency_ms":0,"tool_calls":0,"tool_failures":0,
                               "tool_latency_ms":0,"task_duration_ms":0,"validation_failures":0}},
  "by_stage": {"plan": {"prompt_tokens":0,"completion_tokens":0,"llm_calls":0,
                         "latency_ms":0,"tool_calls":0,"tool_failures":0}},
  "slowest_stages": [{"stage":"...","latency_ms":0,"llm_calls":0,"tokens":0}],
  "bottlenecks": [ ... ]
}
```

排查顺序建议：`slowest_stages` 找瓶颈 → `by_agent` 看谁烧 Token → `events.jsonl` 定位是 LLM 重试、工具失败还是补搜轮次。

## 7. 常见故障排查

| 现象 | 根因 | 处理 |
| --- | --- | --- |
| 启动即报 `未找到 XXX_API_KEY` | provider 与 Key 不匹配 | 在 `.env` 配置对应 Key，或改用 `--provider mock` |
| `证据阶段失败：未获得任何可用证据` | 两路检索都没有可用条目（联网失败 + 知识库为空） | 配置 `TAVILY_API_KEY` / 开启 `--online`；或先 `bootstrap` / `ingest` 导入研报 |
| 任务失败 `输出因 max_tokens 上限被截断` | 模型输出超过 `max_tokens` | 调大 `llm.max_tokens`（当前 8192）；Prompt 已限制证据 ≤8 条、detail ≤120 字，不要放宽这些约束 |
| 证据包为空但日志没有报错 | 历史问题：宽松 schema 静默丢弃截断内容 | 当前已改为 `extra="forbid"` + 截断检测；若复现，检查是否绕过 `EvidenceBundle` 校验 |
| `索引由 hashing-768 构建，当前配置为 zhipu:embedding-3` | 切换 Embedding 后未重建 | 运行 `python -m app.cli bootstrap` 重建索引 |
| `知识库为空` / 检索无结果 | 未建库、阈值过高、语料不相关 | 先 `bootstrap`；下调 `score_threshold`；确认 `data/raw` 有文件 |
| 在线检索无结果但流程继续 | `FallbackSearch` 回退 fixtures | 检查 Tavily Key 与网络；确认不是离线预期行为 |
| `离线模式未找到页面映射：<url>` | `data/fixtures/pages/index.json` 无该 URL | 补充映射与页面文件，或开启 `--online` |
| 报告 `status=partial` | 某个非关键任务失败 | 查看 `run.json.errors`；通常不影响报告产出 |
| Streamlit 页面打不开 | 端口占用 / 未安装 streamlit | 换端口 `--port 8502`；`pip install -r requirements.txt` |
| 智谱 glm-5.x 报 thinking 参数错误 | glm-5.x 不支持 `disabled` | 用 `--thinking low`；代码已自动降级 |
| Windows 下临时目录无法删除 | 日志/事件文件句柄未关闭 | 已在成功与异常路径调用 `shutdown_logging()` / `tracker.close()`；自定义脚本请复用 |
| 前端看不到 zhipu 选项 | 旧版本遗漏 | 当前 `PROVIDER_ENV` 已包含 zhipu；重启 Streamlit |

## 8. 测试

```powershell
python -m unittest discover -s tests -v
```

共 8 个测试文件、36 个测试函数：

| 测试文件 | 覆盖点 |
| --- | --- |
| `test_agent_protocol.py` | ReAct `Action` / `Action Input`（含代码围栏）与 `Final Answer` 解析 |
| `test_structured.py` | JSON 抽取（围栏/前后缀/多候选）、Schema 校验、截断残包拒绝、未知字段拒绝、Markdown 契约 |
| `test_dag.py` | DAG 依赖顺序、并行分支（Barrier 验证真并行）、环检测、失败阻断下游但保留无关分支 |
| `test_tools.py` | 计算器安全性、fixtures 检索、网页正文抽取、工具权限、未知工具、kb_search 元数据、工具描述过滤 |
| `test_splitter.py` | 中文切片长度、重叠、空文本、非法 overlap |
| `test_rag.py` | 入库、检索命中、阈值过滤、持久化与跨实例加载、增量去重、来源列表 |
| `test_eval.py` | 数值/别名/区间匹配、不可回答题拒答、旧格式兼容、Wilson 区间、规则构建器与 8 条不可回答题 |
| `test_pipeline.py` | 端到端 Mock 流水线（报告长度、引用数、章节、证据、指标、HTML、events） |

端到端测试会把索引与 runs 写到临时目录，避免污染真实工作区。

## 9. 扩展指南

### 9.1 新增工具

1. 在 `build_tool_registry()` 中定义函数并 `registry.register(Tool(...))`，提供 JSON Schema 与 `required`；
2. 在 `config/agents.yaml` 对应角色的 `tools` 中加入工具名；
3. 若某任务不应使用，在 `config/tasks.yaml` 的 `tools` 中显式列出允许集合（`[]` 表示禁用）；
4. 工具返回值建议是 JSON 字符串或可 JSON 序列化的对象；异常会被转为 `ok=False` 的 Observation。

### 9.2 新增角色

1. `config/agents.yaml` 增加角色（`role` / `goal` 必填）；
2. 如需自定义行为，在 `app/agents/` 新建子类并覆写 `run()`；
3. 在 `app/agents/factory.py` 的 `_agent_class()` 中注册映射；
4. 在 `config/tasks.yaml` 中创建任务并指向该角色。

### 9.3 新增任务 / 调整 DAG

1. 在 `config/tasks.yaml` 的 `tasks` 列表追加任务；
2. 声明 `depends_on`（Crew 会做存在性、自依赖与环检测）；
3. 指定 `output_schema`（必须是 `SCHEMA_MAP` 中的名字）或 Markdown 契约字段；
4. 若需要新的结构化契约：在 `app/schemas.py` 定义模型并加入 `SCHEMA_MAP`；
5. 若下游需要读取该任务结果，在对应 Agent 的 `_dep_parsed(deps, kind)` 中按 `kind` 查找。

### 9.4 新增 LLM Provider

在 `app/config.py` 的 `PROVIDER_PRESETS` 增加一项（model / base_url / api_key_env / 可选 extra_body），然后在 CLI / UI / `.env` 中即可选择。无需改 `llm.py`。

### 9.5 新增向量库后端

1. 实现 `VectorStore` 子类（`add / search / count / all_ids / reset`，可选 `save / all_chunks`）；
2. `backend` 类属性设为新名称；
3. 在 `create_store()` 中分发；
4. 注意 `search()` 返回的相似度必须“越大越相关”。

### 9.6 新增搜索源

1. 实现 `SearchProvider` 子类，返回 `{title,url,snippet,published_at}`；
2. 在 `build_search_provider()` 中注册；
3. 建议用 `FallbackSearch(主源, fixture)` 包裹，保证离线可演示。

### 9.7 新增 Embedding

1. 实现 `Embedder` 子类（`name`、`dim`、`embed_documents`，输出必须 L2 归一化）；
2. 在 `create_embedder()` 中分发；
3. 切换后必须重建索引，否则 `LocalVectorStore` 会因 embedder 不匹配拒绝加载。

## 10. 生产化注意事项

当前项目定位是可演示、可评测的单机系统。若要上线，至少需要补：

- **并发与队列**：Streamlit 未做任务队列，多用户会串行阻塞；
- **鉴权与配额**：无登录、无用户级 Key 隔离；
- **向量库规模**：本地 NumPy 为全量暴力检索，大规模需换 Chroma / FAISS / 托管向量库；
- **Embedding 质量**：默认 GLM `embedding-3`；离线演示用 Hashing，语义泛化弱，仅作回退；
- **任务幂等与断点续跑**：当前每次 run 独立，无断点恢复；
- **成本控制**：`max_tokens`、`max_react_steps`、`max_coverage_rounds`、`target_chars` 是最直接的旋钮；
- **安全**：`calculator` 已做 AST 白名单，但网页抓取与搜索内容仍应视为不可信输入（Prompt 注入风险），生产环境建议加内容隔离与输出审查。

## 11. 文档维护

- 本 Wiki 描述当前源码状态；修改以下内容时请同步更新文档：
  - `config/*.yaml` 的字段或默认值 → 06
  - `app/schemas.py` 契约 → 03
  - `app/core/crew.py` 调度语义 → 01 / 02
  - `app/rag/*` 检索算法或默认参数 → 04
  - `app/pipeline/evaluator.py` 指标口径 → 05
- 若仅修改实现细节而不改变对外行为，可只在对应章节补充“实现说明”。

返回索引：[README.md](./README.md)
