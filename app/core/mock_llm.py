"""离线 Mock LLM：无 API Key 时提供确定性、可演示的完整流水线输出。

Mock 会真实调用工具（fixtures 检索、页面解析、本地向量库），
只是把“模型推理”替换为基于观察结果的脚本化输出，便于 CI 与离线演示。
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Optional

from app.core.llm import BaseLLM, LLMResponse, estimate_tokens
from app.core.structured import extract_json

_TOPIC_PATTERNS = [
    re.compile(r"研究主题\s*[:：]\s*([^\n]+)"),
    re.compile(r"##\s*研究主题\s*\n+\s*([^\n]+)"),
]
_FOLLOWUP_RE = re.compile(r"补充检索查询\s*[:：]\s*([^\n]+)")


class MockLLM(BaseLLM):
    provider = "mock"

    def __init__(self, settings):
        super().__init__(settings)
        self.model = settings.model or "mock-researcher"
        self._lock = threading.Lock()
        self._coverage_calls = 0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        stage: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        stage = stage or "generic"
        base_stage = stage.split("#")[0]
        handlers = {
            "research_plan": self._plan,
            "web_research": self._web_research,
            "kb_research": self._kb_research,
            "coverage_check": self._coverage,
            "analysis": self._analysis,
            "report_writing": self._write_section,
        }
        handler = handlers.get(base_stage, self._generic)
        text = handler(messages, stage)
        prompt_text = "\n".join(message.get("content", "") for message in messages)
        return LLMResponse(
            text=text,
            prompt_tokens=estimate_tokens(prompt_text),
            completion_tokens=estimate_tokens(text),
            model=self.model,
            latency_ms=0.0,
            finish_reason="stop",
        )

    # ------------------------------------------------------------------ #
    # 兜底
    # ------------------------------------------------------------------ #
    def _generic(self, messages, stage) -> str:
        return "（Mock 模式未实现该阶段，请接入真实模型。）"

    # ------------------------------------------------------------------ #
    # 基础工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def _user_text(messages: list[dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    @staticmethod
    def _all_text(messages: list[dict[str, str]]) -> str:
        return "\n".join(message.get("content", "") for message in messages)

    def _topic(self, messages: list[dict[str, str]]) -> str:
        text = self._all_text(messages)
        for pattern in _TOPIC_PATTERNS:
            match = pattern.search(text)
            if match:
                topic = match.group(1).strip().strip("#- ")
                if topic:
                    return topic[:80]
        return "中国新能源汽车行业（2024）"

    @staticmethod
    def _is_nev(topic: str) -> bool:
        return any(word in topic for word in ("新能源", "汽车", "电动车", "锂电", "动力电池"))

    @staticmethod
    def _observations(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        observations = []
        for message in messages:
            content = message.get("content", "")
            if message.get("role") != "user" or not content.startswith("Observation:"):
                continue
            payload = content[len("Observation:") :].strip()
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                try:
                    data = extract_json(payload)
                except ValueError:
                    continue
            if isinstance(data, dict):
                observations.append(data)
        return observations

    @staticmethod
    def _first_url(observations: list[dict[str, Any]]) -> str:
        for observation in observations:
            results = observation.get("results")
            if isinstance(results, list):
                for row in results:
                    if isinstance(row, dict) and row.get("url"):
                        return str(row["url"])
        return ""

    @staticmethod
    def _action(thought: str, name: str, args: dict[str, Any]) -> str:
        return (
            f"Thought: {thought}\n"
            f"Action: {name}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}"
        )

    @staticmethod
    def _final(payload: dict[str, Any]) -> str:
        return "Final Answer: " + json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 研究规划
    # ------------------------------------------------------------------ #
    def _plan(self, messages, stage) -> str:
        topic = self._topic(messages)
        if self._is_nev(topic):
            queries = [
                ["2024 中国新能源汽车 销量 渗透率", "2024 新能源汽车 市场规模 同比增速"],
                ["2024 新能源汽车 品牌 市场份额 比亚迪", "2024 动力电池 装机量 宁德时代 市占率"],
                ["新能源汽车 产业链 毛利率 价格战", "2024 新能源汽车 补贴退坡 政策 风险"],
            ]
            questions = [
                ("q1", "2024 年中国新能源汽车销量、渗透率与增速如何？", ["销量", "渗透率", "同比增速", "市场规模"]),
                ("q2", "整车与动力电池环节的竞争格局如何？", ["车企份额", "电池份额", "集中度"]),
                ("q3", "产业链关键环节、盈利水平与政策风险如何？", ["毛利率", "产业链环节", "政策风险"]),
            ]
        else:
            queries = [
                [f"{topic} 市场规模 2024", f"{topic} 行业增速 2024"],
                [f"{topic} 竞争格局 主要企业", f"{topic} 市场份额 2024"],
                [f"{topic} 产业链 盈利模式", f"{topic} 政策 风险 2024"],
            ]
            questions = [
                ("q1", f"{topic}的市场规模、增速与驱动因素是什么？", ["市场规模", "增速", "驱动因素"]),
                ("q2", f"{topic}的竞争格局与主要参与者是谁？", ["主要企业", "市场份额"]),
                ("q3", f"{topic}的产业链、盈利模式与风险是什么？", ["产业链", "盈利模式", "风险"]),
            ]
        payload = {
            "topic": topic,
            "objectives": [
                f"回答{topic}的市场规模、增长驱动与竞争格局",
                "识别产业链关键环节、盈利水平与主要风险",
                "形成可供投资/战略决策参考的结论与建议",
            ],
            "sub_questions": [
                {
                    "id": qid,
                    "question": question,
                    "search_queries": queries[index],
                    "data_points": points,
                }
                for index, (qid, question, points) in enumerate(questions)
            ],
            "report_outline": [
                "核心摘要",
                "行业概览与市场规模",
                "竞争格局",
                "产业链与商业模式",
                "趋势与风险",
                "结论与建议",
            ],
            "constraints": ["数据以 2024 年为主", "金额单位为人民币", "离线演示语料为合成数据"],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 联网检索
    # ------------------------------------------------------------------ #
    def _web_research(self, messages, stage) -> str:
        topic = self._topic(messages)
        user_text = self._user_text(messages)
        followup = "补充检索" in user_text or "#followup" in stage
        observations = self._observations(messages)
        search_obs = [
            obs for obs in observations if isinstance(obs.get("results"), list) and "query" in obs
        ]
        fetched = [obs for obs in observations if obs.get("text") and obs.get("url")]
        nev = self._is_nev(topic)

        if followup:
            match = _FOLLOWUP_RE.search(user_text)
            queries = []
            if match:
                queries = [part.strip() for part in re.split(r"[；;]", match.group(1)) if part.strip()]
            query = queries[0] if queries else f"{topic} 政策 风险 2024"
            if not search_obs:
                return self._action("需要补齐政策与风险维度的信息。", "search_web", {"query": query, "max_results": 5})
            if not fetched:
                url = self._first_url(search_obs)
                if url:
                    return self._action("精读一条政策/风险相关结果。", "fetch_page", {"url": url})
            return self._final(self._web_bundle(topic, observations, followup=True))

        if not search_obs:
            query = (
                "2024 中国新能源汽车 销量 渗透率"
                if nev
                else f"{topic} 市场规模 竞争格局 2024"
            )
            return self._action("先检索市场规模与核心数据。", "search_web", {"query": query, "max_results": 6})
        if not fetched:
            url = self._first_url(search_obs)
            if url:
                return self._action("搜索摘要不足以核对口径，读取原始页面。", "fetch_page", {"url": url})
        if len(search_obs) < 2:
            query = (
                "2024 新能源汽车 品牌 市场份额 比亚迪 特斯拉"
                if nev
                else f"{topic} 竞争格局 主要企业 2024"
            )
            return self._action("补充竞争格局信息。", "search_web", {"query": query, "max_results": 6})
        return self._final(self._web_bundle(topic, observations, followup=False))

    def _web_bundle(self, topic: str, observations: list[dict[str, Any]], *, followup: bool) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        queries: list[str] = []
        seen: set[str] = set()
        for observation in observations:
            query = str(observation.get("query", ""))
            if query and query not in queries:
                queries.append(query)
            for row in observation.get("results", []) or []:
                if not isinstance(row, dict) or not row.get("url"):
                    continue
                key = str(row["url"]) + str(row.get("title", ""))[:40]
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "claim": str(row.get("title", "")).strip()[:120],
                        "detail": str(row.get("snippet", ""))[:500],
                        "source_type": "web",
                        "title": str(row.get("title", "")).strip(),
                        "url": str(row.get("url", "")),
                        "published_at": str(row.get("published_at", "")),
                        "confidence": 0.66,
                        "query": query,
                    }
                )
            if observation.get("text") and observation.get("url"):
                key = "fetch:" + str(observation["url"])
                if key not in seen:
                    seen.add(key)
                    items.append(
                        {
                            "claim": f"{observation.get('title', '页面正文')}的关键信息",
                            "detail": str(observation["text"])[:520],
                            "source_type": "web",
                            "title": str(observation.get("title", "")),
                            "url": str(observation["url"]),
                            "published_at": "",
                            "confidence": 0.75,
                            "query": query,
                        }
                    )
        for index, item in enumerate(items, start=1):
            item["id"] = f"E{index}"
        gaps = []
        if not followup:
            gaps.append("政策与风险维度的证据仍需补充")
        if not items:
            gaps.append("联网检索未返回可引用结果")
        return {
            "topic": topic,
            "items": items[:14],
            "gaps": gaps,
            "coverage_notes": "覆盖市场规模、竞争格局与产业链；政策与风险维度需二次检索。"
            if not followup
            else "已补充政策与风险维度信息。",
            "queries": queries,
        }

    # ------------------------------------------------------------------ #
    # 私有研报检索
    # ------------------------------------------------------------------ #
    def _kb_research(self, messages, stage) -> str:
        topic = self._topic(messages)
        nev = self._is_nev(topic)
        observations = self._observations(messages)
        kb_obs = [
            obs
            for obs in observations
            if isinstance(obs.get("results"), list)
            and any(isinstance(row, dict) and "chunk_id" in row for row in obs["results"])
        ]
        kb_calls = sum(
            1
            for message in messages
            if message.get("role") == "assistant"
            and "Action: kb_search" in str(message.get("content", ""))
        )
        queries = (
            ["新能源汽车 销量 渗透率 市场规模", "动力电池 市占率 毛利率 产业链"]
            if nev
            else [f"{topic} 市场规模 财务指标", f"{topic} 竞争格局 毛利率"]
        )
        attempted = max(len(kb_obs), kb_calls)
        if attempted < len(queries):
            return self._action(
                "检索本地研报中的结构化数据。", "kb_search", {"query": queries[attempted]}
            )

        items: list[dict[str, Any]] = []
        used_queries: list[str] = []
        seen: set[str] = set()
        for observation in kb_obs:
            query = str(observation.get("query", ""))
            if query and query not in used_queries:
                used_queries.append(query)
            for row in observation.get("results", []) or []:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                key = f"{row.get('source')}:{row.get('page')}:{text[:30]}"
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    {
                        "claim": text[:70],
                        "detail": text[:520],
                        "source_type": "kb",
                        "title": str(row.get("source", "")),
                        "source": str(row.get("source", "")),
                        "page": row.get("page"),
                        "confidence": min(0.95, float(row.get("score", 0.5)) + 0.2),
                        "query": query,
                    }
                )
        for index, item in enumerate(items, start=1):
            item["id"] = f"E{index}"
        gaps = [] if items else ["本地研报知识库未命中所需数据"]
        return self._final(
            {
                "topic": topic,
                "items": items[:14],
                "gaps": gaps,
                "coverage_notes": "已从本地研报抽取市场规模、财务指标与份额数据。",
                "queries": used_queries,
            }
        )

    # ------------------------------------------------------------------ #
    # 充分性评估
    # ------------------------------------------------------------------ #
    def _coverage(self, messages, stage) -> str:
        with self._lock:
            self._coverage_calls += 1
            call_no = self._coverage_calls
        if call_no == 1:
            payload = {
                "sufficient": False,
                "gaps": ["政策与风险维度证据不足，缺少补贴退坡、价格战与海外贸易壁垒信息"],
                "followup_queries": ["2024 新能源汽车 补贴退坡 价格战 风险"],
                "reason": "市场规模与竞争格局已有证据，但政策风险维度尚未覆盖。",
            }
        else:
            payload = {
                "sufficient": True,
                "gaps": [],
                "followup_queries": [],
                "reason": "补充检索后，市场规模、竞争格局、产业链与政策风险均有可溯源证据。",
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # 结构化分析
    # ------------------------------------------------------------------ #
    def _analysis(self, messages, stage) -> str:
        topic = self._topic(messages)
        items = self._collect_evidence(messages)
        if self._is_nev(topic):
            payload = self._nev_analysis(topic, items)
        else:
            payload = self._generic_analysis(topic, items)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _iter_dicts(text: str):
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj

    def _collect_evidence(self, messages) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in messages:
            for obj in self._iter_dicts(message.get("content", "")):
                rows = obj.get("items")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict) or not row.get("claim"):
                        continue
                    key = (
                        str(row.get("id", ""))
                        + "|"
                        + str(row.get("url", ""))
                        + "|"
                        + str(row.get("source", ""))
                        + "|"
                        + str(row.get("claim", ""))[:40]
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(row)
        return items

    def _nev_analysis(self, topic: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [str(item.get("id")) for item in items if item.get("id")]

        def ref(index: int) -> list[str]:
            if not ids:
                return []
            return [ids[index]] if index < len(ids) else [ids[0]]

        market_ids = ref(0) + ref(1)
        competition_ids = ref(6) + ref(7)
        chain_ids = ref(8) + ref(9)
        trend_ids = ref(2) + ref(3)
        risk_ids = ref(10) + ref(11)
        used = []
        for bucket in (market_ids, competition_ids, chain_ids, trend_ids, risk_ids):
            for item_id in bucket:
                if item_id not in used:
                    used.append(item_id)
        return {
            "topic": topic,
            "executive_summary": (
                "2024 年中国新能源汽车销量达到 1,286.6 万辆，同比增长 35.5%，渗透率升至 40.9%，"
                "行业增长由政策驱动转向消费驱动。与此同时，价格战向整车与电池环节传导，"
                "企业盈利分化加剧，2025 年的核心变量是增速回落后的份额与利润再平衡。"
            ),
            "market": {
                "size": "1,286.6 万辆",
                "growth_rate": "35.5%",
                "cagr": "42%",
                "penetration": "40.9%（批发口径）",
                "forecast": "1,550-1,600 万辆",
                "source_ids": market_ids,
            },
            "key_metrics": [
                {"name": "新能源汽车销量", "value": "1,286.6 万辆", "year": "2024", "note": "批发口径", "source_ids": ref(0)},
                {"name": "渗透率", "value": "40.9%", "year": "2024", "note": "批发口径", "source_ids": ref(1)},
                {"name": "出口量", "value": "128.4 万辆", "year": "2024", "note": "同比增长 6.7%", "source_ids": ref(2)},
                {"name": "动力电池装机量", "value": "548.4 GWh", "year": "2024", "note": "同比增长 41.0%", "source_ids": ref(3)},
                {"name": "公共充电桩保有量", "value": "357.9 万台", "year": "2024", "note": "同比增长 31.0%", "source_ids": ref(4)},
            ],
            "competitive_landscape": [
                {"name": "比亚迪", "share": "33.2%", "positioning": "规模领先，垂直整合与出海并行", "source_ids": ref(6)},
                {"name": "特斯拉中国", "share": "6.0%", "positioning": "高端纯电标杆，面临本土竞品挤压", "source_ids": ref(7)},
                {"name": "吉利汽车", "share": "7.8%", "positioning": "多品牌覆盖，雷神混动放量", "source_ids": ref(8)},
                {"name": "宁德时代", "share": "37.9%（全球动力电池）", "positioning": "技术与规模壁垒，海外扩产", "source_ids": ref(9)},
                {"name": "比亚迪弗迪电池", "share": "16.4%（全球动力电池）", "positioning": "自供为主，成本优势明显", "source_ids": ref(10)},
            ],
            "value_chain": [
                "上游：锂、镍等资源与正负极、隔膜、电解液材料，价格回落缓解成本压力",
                "中游：动力电池、电机、电控与热管理，规模效应与技术迭代决定盈利",
                "下游：整车制造、充电基础设施、后市场服务与电池回收",
            ],
            "trends": [
                {"title": "价格战常态化", "detail": "头部企业以规模换份额，2024 年多轮降价使行业利润率承压。", "source_ids": trend_ids},
                {"title": "智能化与智驾平权", "detail": "高阶辅助驾驶向 15-20 万元价格带下沉，成为差异化竞争点。", "source_ids": ref(2)},
                {"title": "出口与本地化建厂", "detail": "欧洲与东南亚成为重点市场，但关税与合规成本上升。", "source_ids": ref(3)},
                {"title": "电池技术迭代", "detail": "磷酸铁锂、快充与固态电池路线并行，供应链继续降本。", "source_ids": ref(4)},
            ],
            "risks": [
                {"title": "价格战压缩利润", "detail": "整车降价向电池与零部件传导，中小企业现金流承压。", "severity": "高", "source_ids": risk_ids},
                {"title": "补贴退坡与需求透支", "detail": "以旧换新等政策退坡后，部分需求可能被提前透支。", "severity": "中", "source_ids": ref(11)},
                {"title": "海外贸易壁垒", "detail": "欧盟关税与本地化要求提高，出口增速存在不确定性。", "severity": "高", "source_ids": ref(3)},
                {"title": "原材料价格波动", "detail": "碳酸锂价格大幅波动影响电池成本与库存减值。", "severity": "中", "source_ids": ref(4)},
            ],
            "contradictions": [
                "不同机构对 2024 年渗透率的统计口径存在差异（40.9% 与 41.5%），本报告采用中汽协批发口径。",
                "部分来源将出口计入销量，导致市场规模口径不一致，分析时已单独列出出口数据。",
            ],
            "gaps": [
                "2025 年全年数据尚未发布，预测值来自机构指引而非实际值。",
                "二线新势力盈利数据披露不完整，难以做全样本利润率比较。",
            ],
            "confidence": 0.78,
            "evidence_ids": used,
        }

    def _generic_analysis(self, topic: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [str(item.get("id")) for item in items if item.get("id")]
        claims = [str(item.get("claim", "")).strip() for item in items if item.get("claim")]
        summary = (
            f"围绕“{topic}”，本次离线演示共收集 {len(items)} 条证据。"
            + ("主要信息包括：" + "；".join(claims[:3]) + "。" if claims else "知识库与检索均未命中足够证据。")
        )
        return {
            "topic": topic,
            "executive_summary": summary,
            "market": {"size": "未获取", "growth_rate": "未获取", "cagr": "未获取", "penetration": "未获取", "forecast": "未获取", "source_ids": ids[:2]},
            "key_metrics": [],
            "competitive_landscape": [],
            "value_chain": [],
            "trends": [{"title": "证据有限，暂不做趋势外推", "detail": "离线演示模式不对非示例行业做无依据推断。", "source_ids": ids[:2]}],
            "risks": [{"title": "数据不足风险", "detail": "公开检索与本地知识库未覆盖全部关键数据，结论置信度有限。", "severity": "中", "source_ids": ids[:2]}],
            "contradictions": [],
            "gaps": ["该主题不在演示语料范围内，需接入真实检索与研报后重跑"],
            "confidence": 0.4,
            "evidence_ids": ids[:6],
        }

    # ------------------------------------------------------------------ #
    # 章节写作
    # ------------------------------------------------------------------ #
    def _write_section(self, messages, stage) -> str:
        user_text = self._user_text(messages)
        title = self._between(user_text, "SECTION_TITLE") or "章节"
        analysis_text = self._between(user_text, "ANALYSIS_JSON")
        evidence_text = self._between(user_text, "EVIDENCE_JSON")
        try:
            analysis = json.loads(analysis_text) if analysis_text else {}
        except json.JSONDecodeError:
            analysis = {}
        try:
            evidence = json.loads(evidence_text) if evidence_text else []
        except json.JSONDecodeError:
            evidence = []
        topic = str(analysis.get("topic") or self._topic(messages))
        if self._is_nev(topic):
            return self._nev_section(title, analysis, evidence)
        return self._generic_section(title, topic, analysis, evidence)

    @staticmethod
    def _between(text: str, tag: str) -> str:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _cite(ids: list[str]) -> str:
        return "".join(f"[{item}]" for item in ids if item)

    def _nev_section(self, title: str, analysis: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        market = analysis.get("market") or {}
        metrics = analysis.get("key_metrics") or []
        competitors = analysis.get("competitive_landscape") or []
        trends = analysis.get("trends") or []
        risks = analysis.get("risks") or []
        summary = str(analysis.get("executive_summary") or "")
        all_ids = [str(item.get("id")) for item in evidence if item.get("id")] or list(
            analysis.get("evidence_ids") or []
        )

        def ids_of(index: int, count: int = 2) -> list[str]:
            if not all_ids:
                return []
            start = min(index, max(0, len(all_ids) - 1))
            return all_ids[start : start + count]

        cite_market = self._cite(ids_of(0))
        cite_comp = self._cite(ids_of(4))
        cite_chain = self._cite(ids_of(7))
        cite_trend = self._cite(ids_of(2))
        cite_risk = self._cite(ids_of(8))

        if "摘要" in title:
            return f"""## {title}

{summary}

2024 年行业的核心矛盾是“销量高增”与“利润承压”并存：需求端受益于产品力提升、以旧换新与出口扩张，供给端则因产能扩张与价格战导致盈利分化 {cite_market}。从结构看，15-20 万元价格带成为增量主战场，混动与增程车型贡献了重要增量，纯电车型则在补能网络完善的区域加速渗透。

- **市场**：{market.get('size', '未获取')}，同比增长 {market.get('growth_rate', '未获取')}，渗透率 {market.get('penetration', '未获取')}；2025 年销量指引为 {market.get('forecast', '未获取')} {cite_market}。
- **竞争**：比亚迪以 33.2% 的份额领先，特斯拉中国约 6.0%，新势力在细分价格带形成差异化；动力电池环节宁德时代全球份额约 37.9% {cite_comp}。
- **产业链**：上游材料价格回落缓解成本压力，中游电池与电驱加速出海，下游充电与后市场仍是补短板环节 {cite_chain}。

本报告的数据来源分为两类：本地研报（PDF/Word）提供经过整理的销量、份额与财务指标，联网检索提供最新的政策与市场动态。两类证据在合并阶段统一编号，分析阶段进行口径核对。对于来源冲突的数字，报告保留原始口径并说明取舍，避免把不同统计范围的数据直接相加。在方法上，分析团队先用销量、渗透率与出口判断行业景气，再用份额、产能利用率与毛利率判断竞争质量，最后结合政策与贸易环境给出情景假设。该框架数据可得性强、可复用，但部分细分市场缺少统一口径，结论需要随季度数据更新。

综合判断：行业仍处于成长期，但“增量不增利”的风险上升。建议重点关注具备规模效应、技术迭代能力与海外本地化产能的企业，同时跟踪价格战、政策退坡与贸易壁垒的边际变化。"""

        if "市场规模" in title or "概览" in title:
            rows = "\n".join(
                f"| {row.get('name', '')} | {row.get('value', '')} | {row.get('year', '') or '—'} | "
                f"{'、'.join(f'[{i}]' for i in row.get('source_ids', [])) or '—'} |"
                for row in metrics[:6]
                if isinstance(row, dict)
            )
            return f"""## {title}

2024 年中国新能源汽车销量达到 {market.get('size', '未获取')}，同比增长 {market.get('growth_rate', '未获取')}，渗透率提升至 {market.get('penetration', '未获取')}。从节奏看，上半年受价格战与新品周期驱动，下半年在以旧换新政策与购置税减免延续的支撑下保持高景气，月度渗透率多次突破 50% {cite_market}。出口方面，全年出口 128.4 万辆，同比增长 6.7%，增速较 2023 年明显回落，反映海外关税与本地化要求的影响。

| 指标 | 2024 年数值 | 年份/口径 | 证据 |
| --- | --- | --- | --- |
{rows or '| 新能源汽车销量 | 未获取 | 2024 | — |'}

从增长驱动看，2020-2024 年销量 CAGR 约 {market.get('cagr', '未获取')}，行业已从“政策补贴驱动”切换到“产品供给与消费需求驱动”。需要提示的是，不同机构对渗透率与销量口径存在差异：本报告采用中汽协批发口径，若采用零售口径，渗透率与增速会略有不同 {cite_trend}。因此，在做同比比较时应保持口径一致，避免将出口与国内零售直接相加。

分技术路线看，纯电车型仍是主体，但插混与增程在补能条件较弱的地区增长更快；分价格带看，10-20 万元车型贡献主要增量，20 万元以上市场更依赖智能化与品牌力。区域上，一线城市渗透率已处于高位，三四线城市与县域市场是下一阶段增量来源，但其补能网络与售后服务仍在补齐。批发销量包含出口与渠道库存，零售口径更能反映终端需求；若月度数据出现明显偏离，应以累计值而非单月值判断趋势 {cite_market}。

展望 2025 年，机构普遍预计销量达到 {market.get('forecast', '未获取')}，增速回落至 20%-25%。增长动力将从“渗透率快速提升”转向“存量替代与出口”，行业竞争的重点也从“有没有车”转向“成本、智能化与全球化” {cite_market}。"""

        if "竞争" in title:
            rows = "\n".join(
                f"| {row.get('name', '')} | {row.get('share', '')} | {row.get('positioning', '')} | "
                f"{'、'.join(f'[{i}]' for i in row.get('source_ids', [])) or '—'} |"
                for row in competitors[:7]
                if isinstance(row, dict)
            )
            return f"""## {title}

整车端的竞争格局呈现“一超多强”特征。比亚迪凭借垂直整合、规模效应与混动产品矩阵，2024 年份额达到 33.2%，在 10-20 万元价格带形成强定价能力；特斯拉中国份额约 6.0%，主要依靠 Model 3/Y 的品牌与成本优势，但面临本土竞品的持续挤压 {cite_comp}。吉利、长安、奇瑞等传统车企通过多品牌与混动路线快速追赶，理想、蔚来、小鹏等新势力则在 20 万元以上市场以智能化和服务差异化立足。

| 企业 | 份额/规模 | 定位 | 证据 |
| --- | --- | --- | --- |
{rows or '| 比亚迪 | 33.2% | 规模领先 | — |'}

动力电池环节的集中度更高：宁德时代全球市占率约 37.9%，比亚迪弗迪电池约 16.4%，两家合计超过 50%。头部电池企业通过技术迭代、海外建厂与客户绑定巩固壁垒，二线电池企业则在价格战中承压，产能利用率与现金流成为关键变量 {cite_chain}。

从战略上看，头部企业的竞争维度正在从单一价格转向“成本+技术+渠道+全球化”的组合。比亚迪依靠垂直整合控制成本，特斯拉依靠品牌与软件能力维持溢价，传统车企依靠渠道与供应链基础追赶，新势力则在智能化与用户运营上寻找差异。不同路径对应不同的资本开支与盈利节奏，短期份额变化不一定代表长期竞争力。集中度提升也不等于格局稳定：价格战可能加速尾部出清，但也可能使头部企业为保份额而牺牲利润；若海外市场放量，具备本地化产能的企业将获得新的缓冲空间。判断格局是否改善，应同时观察份额、毛利率与现金流，而不是只看销量排名 {cite_comp}。

需要警惕的是，份额数据存在统计口径差异：部分机构按装机量统计，部分按出货量统计；部分榜单将合资品牌单独列出，导致同一企业在不同榜单中的份额不同。本报告在引用时保留原始口径，并在跨来源比较时优先使用同一机构的数据 {cite_comp}。"""

        if "产业链" in title:
            chain = analysis.get("value_chain") or []
            chain_text = "\n".join(f"- {item}" for item in chain) or "- 上游、中游、下游数据未获取"
            return f"""## {title}

新能源汽车产业链可以拆分为上游资源与材料、中游电池与电驱、下游整车与补能服务三个环节。2024 年，碳酸锂价格较 2023 年高点明显回落，缓解了电池成本压力，但也带来库存减值与材料企业盈利下滑；中游电池环节受益于装机量增长，宁德时代等头部企业保持较高产能利用率；下游整车环节则因价格战导致利润率承压 {cite_chain}。

{chain_text}

商业模式上，整车企业正从“卖车一次性收入”转向“硬件+软件+服务”的组合：智能驾驶订阅、充换电服务、电池租赁与后市场维保成为新的利润来源，但短期收入占比仍然较低。电池企业则通过长协、合资建厂与回收业务锁定客户和成本。总体看，产业链利润正在从上游材料向中游头部电池与下游规模化整车企业集中，缺乏技术或规模优势的环节议价能力较弱。

从利润分配看，上游资源品的周期性最强，价格上行时利润集中，价格回落时库存减值压力大；中游电池环节技术壁垒较高，但产能扩张后也面临价格竞争；下游整车环节规模效应显著，却受终端降价与渠道费用挤压。产业链议价能力最终取决于技术不可替代性、客户结构与产能利用率。对投资者而言，判断产业链机会需要区分“量增”与“利增”：装机量、销量增长不必然带来利润增长，毛利率、单位盈利与现金流才是更可靠的指标 {cite_chain}。

从盈利指标看，样本整车企业平均毛利率约 15.6%，头部企业凭借规模与供应链管理保持相对稳定；宁德时代毛利率约 24.4%，显著高于整车环节。价格战若持续，整车与二线电池企业的利润率可能进一步下探，产业链利润分配将继续向头部集中 {cite_chain}。"""

        if "趋势" in title or "风险" in title:
            trend_lines = "\n".join(
                f"{index}. **{row.get('title', '')}**：{row.get('detail', '')}"
                for index, row in enumerate(trends[:4], start=1)
                if isinstance(row, dict)
            )
            risk_lines = "\n".join(
                f"- **{row.get('title', '')}（{row.get('severity', '中')}）**：{row.get('detail', '')}"
                for row in risks[:5]
                if isinstance(row, dict)
            )
            return f"""## {title}

### 关键趋势

{trend_lines or '1. 行业数据不足，暂不做趋势外推。'}

从竞争策略看，头部企业的共同选择是“以规模换成本、以技术换溢价、以出海换增量”。2025 年，智能化将成为 15-20 万元价格带的核心卖点，城市 NOA 与高阶辅助驾驶的渗透率有望快速提升；同时，欧洲、东南亚与中东的本地化产能布局将决定出口增长的持续性 {cite_trend}。

### 主要风险

{risk_lines or '- 数据不足风险：公开资料未覆盖关键变量。'}

上述风险之间存在传导关系：价格战压缩整车利润，进而向上游电池与零部件传导；若海外贸易壁垒同时升级，出口增速下滑会进一步加剧国内产能过剩。建议将“行业利润率、电池价格、出口增速、政策变化”作为四个高频跟踪指标 {cite_risk}。

情景假设方面，基准情形下 2025 年销量增长 20%-25%，价格战延续但边际缓和；乐观情形下，以旧换新与出口超预期，增速接近 30%；悲观情形下，补贴退坡与贸易壁垒叠加，增速降至 15% 以下。三种情景对产业链盈利的影响不同，企业应提前做好产能与现金流预案。风险跟踪上，建议关注月度上险量、渠道库存、电池价格、出口目的地政策与头部企业毛利率；若销量增长与利润率同时下行，说明价格战已进入深水区。"""

        if "结论" in title or "建议" in title:
            return f"""## {title}

综合市场规模、竞争格局与产业链分析，我们对 2025 年中国新能源汽车行业给出三点判断：

1. **增长延续但增速回落**：销量有望达到 1,550-1,600 万辆，渗透率继续提升，但增速较 2024 年明显放缓，行业从“增量竞争”转向“存量替代与结构升级” {cite_market}。
2. **盈利分化加剧**：价格战与规模效应并存，头部整车与电池企业凭借成本、技术与海外产能保持韧性，二线企业面临出清压力 {cite_comp}。
3. **全球化成为第二增长曲线**：出口与海外建厂从“可选项”变为“必选项”，但关税、合规与本地化运营能力将拉开企业差距 {cite_risk}。

基于上述判断，建议不同类型参与者采取差异化策略：整车企业应聚焦爆款车型与成本控制，避免无底线降价；电池与零部件企业应绑定头部客户、加快技术迭代并布局回收；投资者可重点关注具备规模效应、全球化产能与现金流优势的头部企业，同时对政策与贸易摩擦保持敏感。

执行层面，建议建立季度复盘机制：用销量与渗透率判断行业景气，用份额与毛利率判断企业竞争力，用政策与贸易数据判断外生冲击。对研究团队而言，应持续更新本地研报知识库，并把高频数据、公司公告与专家访谈纳入证据链，避免单一来源导致的判断偏差。本报告的结论适用于 2025 年的基准情景，若政策、原材料价格或海外市场发生超预期变化，应重新评估假设并更新模型 {cite_market}。

需要强调的是，本报告基于 2024 年公开数据与演示语料，2025 年实际走势仍取决于政策力度、原材料价格与海外市场变化，结论需要持续跟踪和更新。"""

        return self._generic_section(title, analysis.get("topic", "行业研究"), analysis, evidence)

    def _generic_section(
        self, title: str, topic: str, analysis: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> str:
        claims = [str(item.get("claim", "")).strip() for item in evidence if item.get("claim")]
        ids = [str(item.get("id")) for item in evidence if item.get("id")]
        summary = str(analysis.get("executive_summary") or "")
        cite = self._cite(ids[:3])
        bullets = "\n".join(f"- {claim}" for claim in claims[:4]) or "- 暂无可引用的外部证据。"
        return f"""## {title}

围绕“{topic}”，本节基于当前可获得的 {len(evidence)} 条证据进行梳理。{summary}

主要发现：

{bullets}

需要说明的是，当前运行在离线 Mock 模式，市场规模、份额与财务指标等结构化字段未做无依据推断。接入真实模型与联网检索后，分析师 Agent 会对多源数据做口径核对与交叉验证，撰写员再据此扩写本章节 {cite}。

在方法上，本报告坚持“事实—判断—建议”分层：事实必须有来源，判断需说明依据，建议需标注适用条件。对于证据不足的部分，正文明确写“公开资料未披露”，不以推测替代数据。"""
