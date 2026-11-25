# 03 · Agent 层与数据契约（app/agents、app/schemas.py）

## 1. 角色配置（config/agents.yaml）

| key | role | mode | tools | temperature |
| --- | --- | --- | --- | --- |
| `researcher` | 资深行业研究员 | `react` | `search_web`, `fetch_page`, `kb_search`, `calculator` | 0.2 |
| `analyst` | 行业分析师 | `direct` | `calculator`, `kb_search` | 0.25 |
| `writer` | 资深研究报告撰写人 | `long_form` | `[]` | 0.5 |

加载逻辑在 `load_agents()`：只接受白名单字段（role/goal/backstory/tools/mode/max_steps/temperature），缺少 `role` 或 `goal` 抛 `ConfigError`。

## 2. Agent 工厂（app/agents/factory.py）

### `_agent_class(task, spec)`

```python
if task.kind == "merge_evidence":                       return MergerAgent
if task.kind == "report_writing" or spec.mode == "long_form": return WriterAgent
if spec.key == "researcher":                            return ResearcherAgent
if spec.key == "analyst":                               return AnalystAgent
return Agent
```

优先级说明：

- `merge_evidence` 最高，确保使用确定性实现；
- 写作任务无论角色 key 是什么，只要 kind 或 mode 命中就用 `WriterAgent`；
- 其余按角色 key 映射；
- 未匹配则使用基类 `Agent`（只支持 direct / react）。

### `build_agents(tasks, specs, settings, registry, tracker, llm=None)`

为**每个任务**创建一个 Agent 实例（`agents[task.id] = cls(...)`），使得并行分支之间不共享可变状态（`_usage`、`_tool_calls` 都是实例属性）。

若任务引用了未定义角色，抛 `KeyError`。

## 3. 基类契约回顾

所有 Agent 必须实现：

```python
run(task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult
```

`inputs` 固定包含：

```json
{"topic": "...", "focus": "...", "target_chars": 8000, "today": "2026-11-15"}
```

`deps` 是**直接依赖**的结果字典（key 为 task id），不含间接依赖。

---

## 4. ResearcherAgent（app/agents/researcher.py）

职责：执行一次 ReAct 检索任务，然后做证据充分性质检，不足则补充检索并合并。

### `run()` 主流程

```python
result = super().run(task, deps, inputs)          # 走 run_react
if not task.coverage_check: return result         # collect_kb 不质检
evidence = EvidenceBundle.model_validate(result.parsed or {})
for round in 1..max_coverage_rounds:
    decision = _assess_coverage(...)              # CoverageDecision
    tracker.coverage(...)
    if decision.sufficient:
        evidence.gaps = decision.gaps             # 以最新判断为准
        break
    merge gaps into evidence.gaps
    if not decision.followup_queries: break
    _, parsed = run_react(task, deps, inputs,
                          messages=_followup_messages(...),
                          stage=f"{task.kind}#followup{round}")
    evidence = evidence.merge(EvidenceBundle.model_validate(parsed))
```

最后把 `evidence` 序列化回 `result.text` / `result.parsed`，并同步 usage / tool_calls / retries / duration。

### 质检 `_assess_coverage()`

- 从 `deps` 中找 `kind == "research_plan"` 的结果作为计划文本；
- 构造 payload：`round / topic / evidence.compact(24 条, 320 字) / gaps / queries`；
- system prompt 使用 `COVERAGE` 契约；
- 调用 `_structured_call(..., CoverageDecision, stage="coverage_check", max_retries=1)`；
- 质检本身失败（LLM 异常 / 校验失败）时 **只记录 warning 并保留现有证据**，不中断主流程。

### 补搜 `_followup_messages()`

- 输入：质检发现的 gaps + 最多 3 条 `followup_queries` + 已有证据（20 条、240 字）；
- 明确要求“只输出本轮**新增**证据，最多 6 条，不要重复”；
- 合并由基类 `EvidenceBundle.merge` 去重（见第 8 节）。

### 设计要点

- 补搜轮数受 `pipeline.max_coverage_rounds` 限制（默认 2）。
- 首次质检在 Mock 模式固定返回 `sufficient=false`，用于演示补搜；真实模型按契约判断。
- `collect_web` 开启 `coverage_check: true`，`collect_kb` 未开启。

---

## 5. AnalystAgent（app/agents/analyst.py）

职责：在基类 direct 生成 `AnalysisReport` 之后，做一次**确定性后处理**。

### 后处理逻辑

1. 从依赖中收集合法证据编号：依赖 kind ∈ `{web_research, kb_research, merge_evidence}` 的 `parsed["items"][*]["id"]`；
2. 对以下字段的 `source_ids` 逐项清洗：
   - `market.source_ids`
   - `key_metrics[*].source_ids`
   - `competitive_landscape[*].source_ids`
   - `trends[*].source_ids`
   - `risks[*].source_ids`
3. 删除不存在的编号并记录日志（“移除了 N 个不存在的证据引用”）；
4. 重新计算顶层 `evidence_ids` = 上述所有有效引用的并集，并按数字序排序（`E2 < E10`）；
5. 回写 `result.parsed` 与 `result.text`。

> 这一步保证“分析师编造的 E 编号不会进入报告”，是降低幻觉的运行时闸门。

---

## 6. MergerAgent（app/agents/merger.py）

**不调用 LLM** 的确定性节点。

```python
bundles = [EvidenceBundle.model_validate(r.parsed) for r in deps
           if r.kind in {"web_research", "kb_research"} and 解析成功]
merged = bundles[0].merge(*bundles[1:]) if bundles else EvidenceBundle(topic=...)
merged.topic = inputs["topic"] or merged.topic
merged.reindex()                     # E1..En
if not merged.items:
    raise EvidenceUnavailableError(...)
```

- usage 全为 0，`llm_calls = 0`；
- 返回的 `text` 是格式化 JSON，`parsed` 是 dict；
- **空证据闸门**：两路检索都没有可用条目时直接失败，避免下游生成“无来源报告”。该异常会被 Crew 记录为 `merge_evidence` 失败，Runner 再转成 `PipelineError("证据阶段失败：...")`。

---

## 7. WriterAgent（app/agents/writer.py）

职责：按大纲逐章节生成中文 Markdown 报告，并统一装配指标表、局限与参考资料。

### 7.1 主流程 `run()`

1. 重置 usage / tool_calls / retries（不调用基类，因为不使用其消息装配）。
2. 读取依赖：
   - `research_plan` → `plan`
   - `analysis` → `analysis`
   - 证据：优先 `merge_evidence.parsed`；否则合并 web/kb 两个 bundle。
3. 章节选择：
   - 取 `plan["report_outline"]`；
   - 过滤标题含 `参考 / 附录 / 数据来源说明` 的条目；
   - 最多 8 节；
   - 若不足 4 节，回退到 `DEFAULT_SECTIONS`（核心摘要、行业概览与市场规模、竞争格局、产业链与商业模式、趋势与风险、结论与建议）。
4. 字数预算：`per_section = max(420, target_chars // len(sections))`。
5. 逐章节调用 `_write_section()`。
6. 若全文没有 Markdown 表格，追加 `_metrics_table(analysis)`（优先 `key_metrics`，否则 `competitive_landscape`）。
7. 追加 `_limitations()`（分析 gaps + 证据 gaps + 数据冲突）与 `_references()`（最多 30 条证据）。
8. 生成标题头（主题、生成时间、证据条数、正文字数、免责声明）。
9. 契约校验：

```python
errors = check_markdown(
    report,
    min_chars=contract.min_chars,
    required_sections=contract.required_sections,
    min_citations=min(contract.min_citations, len(evidence.items)),  # 动态下限
    require_table=contract.require_table,
)
if errors: raise AgentOutputError(...)
```

### 7.2 `_write_section()`

- system：`build_writer_system_prompt(spec)`（写作规范，不注入工具）。
- user：报告主题、全文章节清单、当前章节标题（`<SECTION_TITLE>`）、分析 JSON（`<ANALYSIS_JSON>`）、证据清单（`<EVIDENCE_JSON>`，30 条 / 320 字）、写作要求。
- 单节最小字数：`max(150, per_section * 0.3)`。
- 不足则记录 `validation_failed` 并追加“扩写”消息重试，最多 `max_validation_retries` 次。
- 最终若内容 <120 字，抛 `AgentOutputError`。
- `_clean_section()` 会去掉代码围栏；若内容以一级标题 `#` 开头会提升为二级标题 `##`；完全没有标题时补 `## {title}`。

### 7.3 辅助方法

| 方法 | 说明 |
| --- | --- |
| `_dep_parsed(deps, kind)` | 按 kind 查找依赖的 `parsed` |
| `_collect_evidence(deps)` | 优先合并节点，否则合并原始证据 |
| `_metrics_table(analysis)` | 生成“关键指标一览”或“竞争格局一览”表格 |
| `_limitations(analysis, evidence)` | 生成“研究局限与数据缺口” |
| `_references(evidence)` | 生成“参考资料”，URL 优先，KB 证据标注页码 |
| `_has_table(text)` | 检测是否存在表头 + 分隔行 |

### 7.4 为什么按章节写

- 单次生成整篇长文容易触发 `max_tokens` 截断；
- 章节级字数可校验、可重试；
- 所有章节共享同一份分析 JSON 与证据清单，避免前后口径漂移。

---

## 8. 数据契约（app/schemas.py）

### 基类

```python
class _Model(BaseModel):        model_config = ConfigDict(extra="ignore",  str_strip_whitespace=True)
class _StrictModel(BaseModel):  model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
```

- 普通交付物使用 `_Model`，容忍模型多输出字段；
- **证据类使用 `_StrictModel`**，未知字段直接校验失败，防止“字段错位导致静默空包”。

### `SubQuestion` / `ResearchPlan`

```python
SubQuestion: id, question, search_queries[], data_points[]
ResearchPlan: topic, objectives[], sub_questions[], report_outline[], constraints[]
```

### `EvidenceItem` / `EvidenceBundle`

`EvidenceItem` 字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `id` | str | `""` | 合并阶段重写为 `E#` |
| `claim` | str | 必填 | 一句话事实 |
| `detail` | str | `""` | 原文摘要 |
| `source_type` | `"web" \| "kb"` | `"web"` | 来源类型 |
| `title` | str | `""` | 标题 |
| `url` | str | `""` | 网页地址 |
| `source` | str | `""` | 文件名 |
| `page` | int \| None | None | 页码 |
| `published_at` | str | `""` | 发布时间 |
| `confidence` | float | 0.6 | 自动 clamp 到 [0,1] |
| `query` | str | `""` | 产生该证据的检索词 |

`EvidenceBundle` 字段：`topic / items / gaps / coverage_notes / queries`。

关键方法：

| 方法 | 行为 |
| --- | --- |
| `dedupe_key(item)` | `(url 小写 \| source:page)` + `\|` + `claim 去空白前 60 字` |
| `merge(*others)` | 按 dedupe_key 去重合并 items；gaps/queries 去重保序；coverage_notes 用 `；` 拼接；最后 `reindex()` |
| `reindex()` | 按顺序重写 `id = E1..En` |
| `compact(max_items=24, detail_chars=500)` | 生成给 LLM 的紧凑视图（id/claim/detail/item 元信息） |

### `AnalysisReport` 及其子模型

| 模型 | 字段 |
| --- | --- |
| `MarketInfo` | `size, growth_rate, cagr, penetration, forecast, source_ids[]` |
| `Metric` | `name, value, year, note, source_ids[]` |
| `Competitor` | `name, share, positioning, source_ids[]` |
| `Trend` | `title, detail, source_ids[]` |
| `Risk` | `title, detail, severity ∈ {高,中,低}, source_ids[]` |
| `AnalysisReport` | `topic, executive_summary, market, key_metrics[], competitive_landscape[], value_chain[], trends[], risks[], contradictions[], gaps[], confidence, evidence_ids[]` |

`confidence` 同样 clamp 到 [0,1]。

### `CoverageDecision`

```python
sufficient: bool = True
gaps: list[str]
followup_queries: list[str]
reason: str
```

### `SCHEMA_MAP`

```python
{"ResearchPlan": ResearchPlan, "EvidenceBundle": EvidenceBundle,
 "AnalysisReport": AnalysisReport, "CoverageDecision": CoverageDecision}
```

`TaskSpec.output_schema` 通过该表解析为实际模型；找不到时视为“无 schema”（走 Markdown 契约）。

---

## 9. Prompt 体系（app/agents/prompts.py）

### 模板组成

| 常量 | 用途 |
| --- | --- |
| `ROLE_TEMPLATE` | 角色、目标、背景、当前日期 |
| `REACT_PROTOCOL` | 工具清单 + `Thought/Action/Action Input` 与 `Final Answer` 两种输出格式 |
| `DIRECT_PROTOCOL` | 内部 CoT，最终只输出交付物 |
| `RESEARCH_PLAN` | ResearchPlan 的 JSON 契约与硬性要求 |
| `WEB_RESEARCH` | EvidenceBundle（联网）契约 |
| `KB_RESEARCH` | EvidenceBundle（私有研报）契约 |
| `ANALYSIS` | AnalysisReport 契约 |
| `COVERAGE` | CoverageDecision 契约与判断标准 |
| `WRITER_SYSTEM` | 写作规范（只写当前章节、`[E#]` 标注、表格、禁止编造） |

### 组装函数

```python
kind_instruction(kind) -> str
# research_plan / web_research / kb_research / analysis / coverage_check → 对应契约；其他 → ""

build_system_prompt(spec, *, tools_text, instruction, react, today=None) -> str
# ROLE_TEMPLATE + (REACT_PROTOCOL | DIRECT_PROTOCOL) + instruction

build_writer_system_prompt(spec, today=None) -> str
# WRITER_SYSTEM.format(...)
```

### 契约中的关键约束

- **联网**：至少 2 次 `search_web`、至少 1 次 `fetch_page`；冲突数据保留两条并注明来源。
- **知识库**：至少 2 组 `kb_search`；保留文件名与页码。
- **分析**：只能引用真实存在的 `E#`；无证据写“未获取”；派生指标写清口径。
- **质检**：每个子问题至少 1 条可溯源证据；补充检索词不得重复已有 queries。

---

## 10. 任务描述模板（config/tasks.yaml）

`TaskSpec.render_description(inputs)` 使用 Jinja2 渲染 `description`：

- 可用变量：`topic`、`focus`、`target_chars`、`today`；
- `focus` 通过 `safe_inputs.setdefault("focus", "")` 保证未传时也能渲染；
- 渲染后 `strip()`，作为 user prompt 的任务段落。

新增任务时，若 description 中引用了不存在的变量，Jinja2 会抛异常；只使用上述四个变量最安全。

---

## 11. 错误与失败语义

| 场景 | 异常 | 传播结果 |
| --- | --- | --- |
| 结构化输出重试后仍不合规 | `AgentOutputError` | Crew 记录任务失败，下游跳过 |
| ReAct 达到步数/重试上限 | `AgentOutputError` | 同上 |
| Markdown 契约不满足（writer） | `AgentOutputError` | 同上 |
| 合并没有任何证据 | `EvidenceUnavailableError` | Runner 转 `PipelineError` |
| 质检失败 | 仅 warning | 保留现有证据，继续 |
| 补搜失败 | 仅 warning | 保留现有证据，结束补搜 |
| 工具失败 | `ToolCallResult(ok=False)` | 作为 Observation 回喂模型 |

继续阅读：[04-rag.md](./04-rag.md)
