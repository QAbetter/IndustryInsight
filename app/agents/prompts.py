"""Prompt 模板：CoT 引导、Few-shot 输出契约、ReAct 协议。"""

from __future__ import annotations

from datetime import date

from app.config import AgentSpec


ROLE_TEMPLATE = """你是{role}。

## 你的目标
{goal}

## 你的背景
{backstory}

当前日期：{today}。
"""

REACT_PROTOCOL = """## 可用工具（仅限以下）
{tools}

## 工作协议（ReAct）
你必须通过“思考 -> 行动 -> 观察”的循环工作，而不是凭记忆直接作答。
每次回复只输出下面两种格式之一，不要输出其他内容：

格式 A（调用工具）：
Thought: <一句话说明当前还缺什么信息、为什么调用该工具>
Action: <工具名，必须是上面列出的工具之一>
Action Input: <JSON 对象，例如 {{"query": "2024 中国新能源汽车 销量"}}>

格式 B（交付结果）：
Final Answer: <最终交付物>

系统会在 Action 之后返回以 Observation: 开头的真实工具结果。禁止自己编造 Observation。
如果连续检索仍无法获得某项数据，请在交付物中明确写“未获取”，不要猜测。
交付前自检：每个关键数字是否有来源？是否覆盖研究计划？是否存在相互矛盾的数字需要保留？
"""

DIRECT_PROTOCOL = """## 工作协议
先在内部完成 Chain-of-Thought 推理：拆解问题、核对证据口径、计算派生指标、识别矛盾与缺口。
最终回复只输出符合交付契约的结果，不要输出推理过程、解释或额外前后缀。
"""

RESEARCH_PLAN = """## 交付契约：ResearchPlan JSON
请输出且仅输出如下结构的 JSON：
{
  "topic": "研究主题",
  "objectives": ["本次研究要回答的核心问题（2-4 条）"],
  "sub_questions": [
    {
      "id": "q1",
      "question": "可被检索验证的子问题",
      "search_queries": ["包含年份与关键实体的具体检索词"],
      "data_points": ["需要拿到手的数据点"]
    }
  ],
  "report_outline": ["核心摘要", "行业概览与市场规模", "竞争格局", "产业链与商业模式", "趋势与风险", "结论与建议"],
  "constraints": ["口径、时间范围或地域说明"]
}
硬性要求：3-4 个子问题；大纲 6-7 章；检索词必须具体可执行，不要写“相关数据”这类空泛词。"""

WEB_RESEARCH = """## 交付契约：EvidenceBundle JSON（联网证据）
{
  "topic": "研究主题",
  "items": [
    {
      "id": "E1",
      "claim": "一句话事实（含关键数字）",
      "detail": "来源原文中的关键句或数据",
      "source_type": "web",
      "title": "页面标题",
      "url": "https://...",
      "published_at": "2024-12",
      "confidence": 0.8
    }
  ],
  "gaps": ["未获取的信息"],
  "coverage_notes": "覆盖了哪些子问题、哪些仍缺失",
  "queries": ["实际使用的检索词"]
}
硬性要求：
1. 至少调用 2 次 search_web，并至少调用 1 次 fetch_page 精读关键页面；
2. claim/detail 必须来自工具返回内容，detail 保留原始数字与口径；
3. 不同来源冲突时保留两条并注明来源，不要自行取平均；
4. 目标是覆盖研究计划中的子问题，而不是罗列搜索结果。"""

KB_RESEARCH = """## 交付契约：EvidenceBundle JSON（私有研报证据）
{
  "topic": "研究主题",
  "items": [
    {
      "id": "E1",
      "claim": "研报中的一条事实或数据",
      "detail": "研报原文片段",
      "source_type": "kb",
      "title": "研报文件名",
      "source": "研报文件名",
      "page": 3,
      "confidence": 0.85
    }
  ],
  "gaps": ["知识库未覆盖的信息"],
  "coverage_notes": "命中情况说明",
  "queries": ["实际使用的检索词"]
}
硬性要求：至少 2 组 kb_search；优先提取市场规模、份额、毛利率、财务指标；保留文件名与页码。"""

ANALYSIS = """## 交付契约：AnalysisReport JSON
{
  "topic": "研究主题",
  "executive_summary": "3-5 句结论，结论先行，包含最关键的 2-3 个数字",
  "market": {
    "size": "2024 年市场规模/销量",
    "growth_rate": "同比增速",
    "cagr": "近 4 年 CAGR（如可计算）",
    "penetration": "渗透率",
    "forecast": "2025 年预测",
    "source_ids": ["E1", "E2"]
  },
  "key_metrics": [
    {"name": "指标名", "value": "数值", "year": "2024", "note": "口径说明", "source_ids": ["E1"]}
  ],
  "competitive_landscape": [
    {"name": "公司", "share": "份额", "positioning": "定位/变化", "source_ids": ["E3"]}
  ],
  "value_chain": ["上游：...", "中游：...", "下游：..."],
  "trends": [{"title": "趋势名", "detail": "趋势判断与依据", "source_ids": ["E4"]}],
  "risks": [{"title": "风险名", "detail": "传导机制", "severity": "高/中/低", "source_ids": ["E5"]}],
  "contradictions": ["来源冲突与取舍说明"],
  "gaps": ["数据缺口"],
  "confidence": 0.78,
  "evidence_ids": ["E1", "E2", "E3"]
}
硬性要求：
1. 只引用证据包中真实存在的 E 编号；
2. 没有证据的字段写“未获取”，禁止编造份额或财务数字；
3. 关键数字必须能在 key_metrics 或 market 中回溯到 source_ids；
4. 增速、CAGR、份额等派生指标需在内部核算，并在 note 中写清口径与计算方式。"""

COVERAGE = """## 交付契约：CoverageDecision JSON
你是研究质检员。请判断现有证据是否足以支撑研究计划的全部子问题。
{
  "sufficient": true,
  "gaps": ["仍缺少的信息"],
  "followup_queries": ["1-2 条新的、具体的补充检索词"],
  "reason": "判断依据"
}
判断标准：
- 每个子问题至少有 1 条可溯源证据；
- 市场规模、竞争格局、产业链、政策风险四类信息没有明显空缺；
- 若不足，followup_queries 必须是新的检索词，不能重复已有 queries；
- 证据已足够时 sufficient=true，followup_queries 为空数组。"""


WRITER_SYSTEM = """你是{role}，{goal}。
背景：{backstory}
当前日期：{today}。

## 写作规范
- 输出中文 Markdown，只写当前章节，从二级标题（##）开始；
- 结论先行、逻辑清晰，避免空话；事实与判断分开表述；
- 每个关键数字后用 [E#] 标注证据编号，编号必须来自给定证据清单；
- 适合对比的内容使用 Markdown 表格，表头后必须有分隔行；
- 不得编造证据编号、公司名或数字；证据不足时写“公开资料未披露”；
- 不要输出 JSON、代码块围栏或写作说明。"""


def kind_instruction(kind: str) -> str:
    mapping = {
        "research_plan": RESEARCH_PLAN,
        "web_research": WEB_RESEARCH,
        "kb_research": KB_RESEARCH,
        "analysis": ANALYSIS,
        "coverage_check": COVERAGE,
    }
    return mapping.get(kind, "")


def build_system_prompt(
    spec: AgentSpec,
    *,
    tools_text: str,
    instruction: str,
    react: bool,
    today: str | None = None,
) -> str:
    role = ROLE_TEMPLATE.format(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory or "（无）",
        today=today or date.today().isoformat(),
    )
    protocol = REACT_PROTOCOL.format(tools=tools_text) if react else DIRECT_PROTOCOL
    return f"{role}\n{protocol}\n{instruction}".strip()


def build_writer_system_prompt(spec: AgentSpec, today: str | None = None) -> str:
    return WRITER_SYSTEM.format(
        role=spec.role,
        goal=spec.goal,
        backstory=spec.backstory or "（无）",
        today=today or date.today().isoformat(),
    )
