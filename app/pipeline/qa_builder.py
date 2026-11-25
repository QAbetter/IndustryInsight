"""评测集构建器。

两种模式：
1. 真实 LLM：逐切片生成 grounded QA（问题、标准答案、原文证据片段）；
2. 离线模式：规则抽取 + 人工种子题合并，保证无 API Key 也能构建、校验评测集。

正式用于简历前，请人工抽检 evidence_quote，并建议扩到 60-100 题。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from app.core.llm import BaseLLM
from app.core.structured import extract_json
from app.pipeline.eval_schema import normalize_qa_row
from app.rag.knowledge import KnowledgeBase
from app.rag.metadata import infer_source_context
from app.rag.models import Chunk

logger = logging.getLogger("insight.qa_builder")

COMPANIES = [
    "比亚迪弗迪电池",
    "比亚迪弗迪",
    "比亚迪",
    "特斯拉中国",
    "特斯拉",
    "吉利汽车",
    "吉利",
    "长安汽车",
    "长安",
    "奇瑞汽车",
    "奇瑞",
    "理想汽车",
    "理想",
    "蔚来",
    "小鹏",
    "宁德时代",
    "LG 新能源",
    "LG新能源",
    "中创新航",
    "亿纬锂能",
]
BATTERY_COMPANIES = {"宁德时代", "比亚迪弗迪电池", "比亚迪弗迪", "LG 新能源", "LG新能源", "中创新航", "亿纬锂能"}

UNIT_RE = r"(?:万辆|万台|GWh|亿元|万元|个百分点|%)"
NUMBER_UNIT_RE = re.compile(rf"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>{UNIT_RE})?")
RANGE_RE = re.compile(r"(?P<low>\d[\d,]*(?:\.\d+)?)\s*[-~至]\s*(?P<high>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>万辆|%)")
PERCENT_RE = re.compile(r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*%")
PERCENT_RANGE_RE = re.compile(
    r"(?P<low>\d[\d,]*(?:\.\d+)?)\s*%\s*[-~至]\s*(?P<high>\d[\d,]*(?:\.\d+)?)\s*%"
)

UNANSWERABLE = [
    {
        "question": "2025 年埃斯顿人形机器人出货量是多少台？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "easy",
        "answerable": False,
    },
    {
        "question": "汇川技术 2025 年上半年工业机器人业务收入是多少？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "easy",
        "answerable": False,
    },
    {
        "question": "拓斯达 2024 年协作机器人销量是多少台？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "medium",
        "answerable": False,
    },
    {
        "question": "2025 年中国工业机器人平均单价是多少万元？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "medium",
        "answerable": False,
    },
    {
        "question": "新时达 2024 年机器人业务毛利率是多少？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "medium",
        "answerable": False,
    },
    {
        "question": "2024 年中国市场 SCARA 机器人销量是多少台？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "medium",
        "answerable": False,
    },
    {
        "question": "埃斯顿 2025 年第一季度营业收入是多少？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "easy",
        "answerable": False,
    },
    {
        "question": "2030 年中国工业机器人市场规模预测是多少亿元？",
        "gold_answer": "无法确定",
        "answer_aliases": ["无法确定", "未提供", "未披露"],
        "category": "unanswerable",
        "difficulty": "medium",
        "answerable": False,
    },
]

# 年报里的公司治理/文档结构类内容，不适合作为行业调研评测题
LOW_VALUE_QUESTION_RE = re.compile(
    r"目录|封面|第几页|第几节|页码|起始页|网址|网站|会计师事务所|保荐机构|持续督导|公告编号"
    r"|证券代码|股票代码|公司全称|全称是什么|报告全文|备置地点|签字|盖章|法定代表人|董事会秘书"
    r"|联系方式|邮政编码|传真|注册地址|办公地址|股票简称"
)
LOW_VALUE_CHUNK_RE = re.compile(
    r"监事会|董事会决议|股东大会|独立董事|内部控制|保荐|审计报告|财务报表附注"
    r"|合并资产负债表|母公司资产负债表|合并所有者权益变动表"
)
VALUABLE_CHUNK_RE = re.compile(
    r"营业收入|营业总收入|净利润|毛利率|毛利|研发投入|研发费用|出货|销量|产量|市场份额|市占率"
    r"|同比|增长率|产能|在手订单|客户|市场规模|竞争格局|现金流|资产负债率"
    r"|机器人|减速器|伺服|控制器|本体|revenue|market share|instal|units|growth|forecast|CAGR",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text).replace(",", "").replace("，", "")).lower()


def _sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。；;])\s*|\n+", text)
    return [piece.strip(" \t-|") for piece in pieces if len(piece.strip()) >= 12]


def _first_fact(sentence: str) -> Optional[tuple[str, str]]:
    for match in NUMBER_UNIT_RE.finditer(sentence):
        unit = match.group("unit") or ""
        number = match.group("number")
        if unit:
            return number, unit
    return None


def _percent_fact(sentence: str) -> Optional[str]:
    match = PERCENT_RE.search(sentence)
    return f"{match.group('number')}%" if match else None


def _number_near(sentence: str, phrase: str) -> Optional[tuple[str, str]]:
    if not phrase or phrase not in sentence:
        return None
    position = sentence.find(phrase)
    best: Optional[tuple[int, str, str]] = None
    for match in NUMBER_UNIT_RE.finditer(sentence):
        unit = match.group("unit") or ""
        if not unit:
            continue
        distance = abs(match.start() - position)
        if best is None or distance < best[0]:
            best = (distance, match.group("number"), unit)
    return (best[1], best[2]) if best else None


def _percent_near(sentence: str, phrase: str) -> Optional[str]:
    if not phrase or phrase not in sentence:
        return None
    position = sentence.find(phrase)
    best: Optional[tuple[int, str]] = None
    for match in PERCENT_RE.finditer(sentence):
        distance = abs(match.start() - position)
        if best is None or distance < best[0]:
            best = (distance, match.group("number"))
    return f"{best[1]}%" if best else None


def _percent_range_fact(sentence: str) -> Optional[str]:
    match = PERCENT_RANGE_RE.search(sentence)
    if not match:
        return None
    return f"{match.group('low')}%-{match.group('high')}%"


def _range_fact(sentence: str) -> Optional[str]:
    match = RANGE_RE.search(sentence)
    if not match:
        return None
    return f"{match.group('low')}-{match.group('high')} {match.group('unit')}"


def _company(sentence: str) -> str:
    for name in COMPANIES:
        if name in sentence:
            return name
    return ""


def _category(sentence: str) -> str:
    if any(word in sentence for word in ("关税", "免税额", "减税额", "购置税", "补贴", "双积分")):
        return "policy"
    if "充电" in sentence:
        return "charging"
    if "装机量" in sentence or "市占率" in sentence:
        return "battery"
    if "毛利率" in sentence or "利润率" in sentence:
        return "profit"
    if any(word in sentence for word in ("份额", "竞争格局", "销量排名")):
        return "competition"
    if "2025" in sentence and ("预测", "预计", "展望", "增速" ) and "销量" in sentence:
        return "outlook"
    return "market"


def _make_row(
    *,
    question: str,
    gold_answer: str,
    evidence_quote: str,
    chunk: Chunk,
    category: str,
    difficulty: str = "easy",
    requires_calc: bool = False,
) -> dict[str, Any]:
    aliases = [gold_answer]
    bare = gold_answer.replace(",", "").split(" ")[0]
    if bare and bare not in aliases:
        aliases.append(bare)
    return {
        "question": question,
        "gold_answer": gold_answer,
        "answer_aliases": aliases,
        "evidence_quote": evidence_quote[:240],
        "source": chunk.source,
        "page": chunk.page,
        "category": category,
        "difficulty": difficulty,
        "answerable": True,
        "requires_calc": requires_calc,
    }


def ground_question(question: str, source: str) -> str:
    """把“2024年营业收入是多少？”补全为“在埃斯顿2024年年报中，营业收入是多少？”。"""
    context = infer_source_context(source)
    text = str(question or "").strip()
    if not context or not text:
        return text
    if context["company"] == "IFR":
        if "IFR" in text or "World Robotics" in text:
            return text
    else:
        # 已经同时包含公司名和年份的题目无需重复加前缀
        if context["company"] in text and re.search(r"20\d{2}\s*年", text):
            return text
        # 去掉与报告年份重复的开头年份，避免“2024年年报中，2024年…”
        text = re.sub(rf"^{context['report_year']}\s*年\s*", "", text)
        if text.startswith("在") and "年报" in text[:24]:
            return text
    return f"{context['prefix']}{text}"


def ground_row(row: dict[str, Any]) -> dict[str, Any]:
    """给单条 QA 补全公司/年份，并写入可过滤的元数据。"""
    context = infer_source_context(row.get("source", ""))
    if not context:
        return row
    grounded = dict(row)
    grounded["question"] = ground_question(row.get("question", ""), row.get("source", ""))
    grounded["company"] = context["company"]
    grounded["report_year"] = context["report_year"]
    grounded["report_name"] = context["report_name"]
    return grounded


def _heuristic_row(sentence: str, chunk: Chunk) -> Optional[dict[str, Any]]:
    text = sentence
    fact = _first_fact(text)
    percent = _percent_fact(text)
    range_fact = _range_fact(text)
    percent_range = _percent_range_fact(text)

    if ("复合增长率" in text or "CAGR" in text) and percent:
        return _make_row(
            question="2020-2024 年新能源汽车销量的复合增长率是多少？",
            gold_answer=percent,
            evidence_quote=text,
            chunk=chunk,
            category="market",
            difficulty="medium",
        )
    if "免税额" in text:
        near = _number_near(text, "免税额") or fact
        if near:
            return _make_row(
                question="2024-2025 年新能源汽车每辆免税额上限是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="policy",
                difficulty="medium",
            )
    if "减税额" in text:
        near = _number_near(text, "减税额") or fact
        if near:
            return _make_row(
                question="2026-2027 年新能源汽车每辆减税额上限是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="policy",
                difficulty="medium",
            )
    if "关税" in text and percent:
        return _make_row(
            question="欧盟对中国电动车加征的最高关税大约是多少？",
            gold_answer=_percent_near(text, "关税") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="policy",
            difficulty="medium",
        )
    if "私人充电桩" in text:
        near = _number_near(text, "私人充电桩") or fact
        if near:
            return _make_row(
                question="2024 年私人充电桩数量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="charging",
            )
    if "充电基础设施" in text and ("同比" in text or "增长" in text) and percent:
        return _make_row(
            question="2024 年充电基础设施累计数量的同比增速是多少？",
            gold_answer=_percent_near(text, "充电基础设施") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="charging",
            difficulty="medium",
        )
    if "充电基础设施" in text:
        near = _number_near(text, "充电基础设施") or fact
        if near:
            return _make_row(
                question="2024 年充电基础设施累计数量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="charging",
            )
    if "公共充电桩" in text and ("同比" in text or "增长" in text) and percent:
        return _make_row(
            question="2024 年公共充电桩保有量的同比增速是多少？",
            gold_answer=_percent_near(text, "公共充电桩") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="charging",
            difficulty="medium",
        )
    if "公共充电桩" in text:
        near = _number_near(text, "公共充电桩") or fact
        if near:
            return _make_row(
                question="2024 年公共充电桩保有量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="charging",
            )
    if "2025" in text and "销量" in text and "增速" in text and percent_range:
        return _make_row(
            question="机构预计 2025 年新能源汽车销量的增速区间是多少？",
            gold_answer=percent_range,
            evidence_quote=text,
            chunk=chunk,
            category="outlook",
            difficulty="medium",
        )
    if "2025" in text and "销量" in text and range_fact:
        return _make_row(
            question="机构预计 2025 年新能源汽车销量是多少？",
            gold_answer=range_fact,
            evidence_quote=text,
            chunk=chunk,
            category="outlook",
            difficulty="medium",
        )
    if "装机量" in text:
        scope = "全球" if "全球" in text else "中国"
        near = _number_near(text, f"{scope}动力电池装机量") or fact
        if near:
            return _make_row(
                question=f"2024 年{scope}动力电池装机量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="battery",
            )
    if ("市占率" in text or "份额" in text) and percent:
        company = _company(text)
        if not company:
            return None
        metric = "全球市占率" if "全球" in text and "市占率" in text else ("市场份额" if "份额" in text else "市占率")
        category = "battery" if company in BATTERY_COMPANIES else "competition"
        return _make_row(
            question=f"{company}的{metric}是多少？",
            gold_answer=_percent_near(text, company) or percent,
            evidence_quote=text,
            chunk=chunk,
            category=category,
        )
    if "毛利率" in text and percent:
        if "宁德时代" in text:
            subject = "宁德时代"
            value = _percent_near(text, "宁德时代") or percent
        elif "样本整车" in text:
            subject = "样本整车企业"
            value = _percent_near(text, "样本整车") or percent
        else:
            subject = "行业"
            value = _percent_near(text, "毛利率") or percent
        category = "battery" if subject == "宁德时代" else "profit"
        return _make_row(
            question=f"{subject}的毛利率是多少？",
            gold_answer=value,
            evidence_quote=text,
            chunk=chunk,
            category=category,
        )
    if "利润率" in text and percent:
        return _make_row(
            question="2024 年汽车行业整体利润率是多少？",
            gold_answer=_percent_near(text, "利润率") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="profit",
        )
    if "渗透率" in text and percent:
        return _make_row(
            question="2024 年新能源汽车渗透率是多少？",
            gold_answer=_percent_near(text, "渗透率") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="market",
        )
    if "出口" in text and percent and ("同比" in text or "增长" in text):
        return _make_row(
            question="2024 年新能源汽车出口的同比增速是多少？",
            gold_answer=_percent_near(text, "出口") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="market",
            difficulty="medium",
        )
    if "出口" in text:
        near = _number_near(text, "出口") or fact
        if near:
            return _make_row(
                question="2024 年新能源汽车出口量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="market",
            )
    if "销量" in text and percent and ("同比" in text or "增长" in text):
        return _make_row(
            question="2024 年中国新能源汽车销量的同比增速是多少？",
            gold_answer=_percent_near(text, "同比") or percent,
            evidence_quote=text,
            chunk=chunk,
            category="market",
            difficulty="medium",
        )
    if "销量" in text:
        near = _number_near(text, "销量") or fact
        if near:
            return _make_row(
                question="2024 年中国新能源汽车销量是多少？",
                gold_answer=f"{near[0]} {near[1]}",
                evidence_quote=text,
                chunk=chunk,
                category="market",
            )
    return None


def _table_rows(chunk: Chunk) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in chunk.text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or "---" in cells[1]:
            continue
        company = cells[0]
        share = cells[1]
        if company not in COMPANIES or not re.fullmatch(r"\d+(?:\.\d+)?%", share):
            continue
        category = "battery" if company in BATTERY_COMPANIES else "competition"
        metric = "全球市占率" if category == "battery" else "市场份额"
        rows.append(
            _make_row(
                question=f"{company}的{metric}是多少？",
                gold_answer=share,
                evidence_quote=line,
                chunk=chunk,
                category=category,
            )
        )
    return rows


def heuristic_candidates(chunks: list[Chunk], per_chunk: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        table_rows = _table_rows(chunk)
        rows.extend(table_rows[:per_chunk])
        count = 0
        for sentence in _sentences(chunk.text):
            if count >= per_chunk:
                break
            row = _heuristic_row(sentence, chunk)
            if row:
                rows.append(row)
                count += 1
    return rows


def _llm_candidates(llm: BaseLLM, chunks: list[Chunk], per_chunk: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        prompt = (
            "你是行业研报评测数据标注员。阅读片段，生成最多 "
            f"{per_chunk} 个可以被片段直接回答的问题，覆盖不同事实。\n"
            "只输出 JSON 数组，每项字段：\n"
            '{"question":"", "gold_answer":"", "answer_aliases":[""], '
            '"evidence_quote":"必须是片段原文子串", '
            '"category":"financial|competition|market|product|policy|outlook", '
            '"difficulty":"easy|medium|hard", "requires_calc":false}\n'
            "规则：问题不能包含答案；数字保留单位；evidence_quote 必须逐字出现在片段中；"
            "不要出需要外部知识或研报未披露的问题；不要重复；"
            "只针对业务与经营数据出题（营收、利润、毛利率、出货量、市占率、研发、产能、行业趋势），"
            "禁止出目录、页码、公司全称、网址、会计师事务所、保荐机构等文档结构或公司治理类问题。\n\n"
            f"片段来源：{chunk.source} 第 {chunk.page or '-'} 页\n"
            f"片段：\n{chunk.text}"
        )
        try:
            response = llm.chat(
                [
                    {"role": "system", "content": "你只输出合法 JSON，不要输出解释。"},
                    {"role": "user", "content": prompt},
                ],
                stage="qa_build",
                temperature=0.2,
            )
            payload = extract_json(response.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 生成 QA 失败（%s）：%s", chunk.source, exc)
            continue
        items = payload.get("items") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            continue
        for item in items[:per_chunk]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            gold = str(item.get("gold_answer", "")).strip()
            quote = str(item.get("evidence_quote", "")).strip()
            if not question or not gold or not quote:
                continue
            if _normalize(quote[:40]) not in _normalize(chunk.text):
                logger.debug("丢弃证据片段不匹配的 QA：%s", question)
                continue
            rows.append(
                {
                    "question": question,
                    "gold_answer": gold,
                    "answer_aliases": item.get("answer_aliases") or [gold],
                    "evidence_quote": quote,
                    "source": chunk.source,
                    "page": chunk.page,
                    "category": item.get("category") or _category(quote),
                    "difficulty": item.get("difficulty") or "medium",
                    "answerable": True,
                    "requires_calc": bool(item.get("requires_calc", False)),
                }
            )
    return rows


def _load_rows(path: Optional[Path], generator: str) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = normalize_qa_row(json.loads(line))
        row["_generator"] = generator
        rows.append(row)
    return rows


def _enrich_legacy(rows: list[dict[str, Any]], chunks: list[Chunk]) -> None:
    for row in rows:
        if row.get("evidence_quote") or not row.get("answerable", True):
            continue
        keywords = row.get("answer_keywords") or [row.get("gold_answer", "")]
        for chunk in chunks:
            for sentence in _sentences(chunk.text):
                if any(keyword and _normalize(keyword) in _normalize(sentence) for keyword in keywords):
                    row["evidence_quote"] = sentence[:240]
                    row["source"] = chunk.source
                    row["page"] = chunk.page
                    break
            if row.get("evidence_quote"):
                break


def _number_set(text: str) -> set[str]:
    values = set()
    for item in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text)):
        raw = item.replace(",", "")
        try:
            number = float(raw)
            values.add(str(int(number)) if number.is_integer() else f"{number:g}")
        except ValueError:
            values.add(raw)
    return values


def _question_key(question: str) -> str:
    key = re.sub(r"[\s，。？?、]", "", question)
    for word in ("是多少", "多少", "的", "请问", "根据", "研报", "数据显示"):
        key = key.replace(word, "")
    return key


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_numeric: list[tuple[frozenset[str], str]] = []
    seen_text: set[tuple[str, str]] = set()
    for row in rows:
        question_key = _question_key(row["question"])
        if question_key in seen_questions:
            continue
        answerable = row.get("answerable", True)
        category = row.get("category", "fact")
        numbers = _number_set(row.get("gold_answer", "")) if answerable else set()
        if answerable:
            if numbers:
                duplicate = any(
                    existing_category == category
                    and (numbers <= existing or existing <= numbers)
                    for existing, existing_category in seen_numeric
                )
                if duplicate:
                    continue
            else:
                text_key = (_normalize(row.get("gold_answer", "")), category)
                if text_key in seen_text:
                    continue
                seen_text.add(text_key)
        seen_questions.add(question_key)
        if answerable and numbers:
            seen_numeric.append((frozenset(numbers), category))
        result.append(row)
    return result


def _validate_rows(rows: list[dict[str, Any]], chunks: list[Chunk]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("answerable", True):
            valid.append(row)
            continue
        if not row.get("gold_answer"):
            continue
        quote = _normalize(row.get("evidence_quote", ""))[:80]
        if quote and any(quote[:40] in _normalize(chunk.text) for chunk in chunks):
            valid.append(row)
            continue
        # 兜底：按标准答案反查一个证据片段
        answer_numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", row.get("gold_answer", ""))
        found = False
        for chunk in chunks:
            for sentence in _sentences(chunk.text):
                if any(number and number.replace(",", "") in sentence.replace(",", "") for number in answer_numbers):
                    row["evidence_quote"] = sentence[:240]
                    row["source"] = chunk.source
                    row["page"] = chunk.page
                    valid.append(row)
                    found = True
                    break
            if found:
                break
        if not found:
            logger.warning("丢弃无法回溯证据的 QA：%s", row.get("question"))
    return valid


def _is_low_value_question(question: str) -> bool:
    return bool(LOW_VALUE_QUESTION_RE.search(str(question)))


def _is_valuable_chunk(text: str) -> bool:
    return bool(VALUABLE_CHUNK_RE.search(text)) and not LOW_VALUE_CHUNK_RE.search(text)


def _evenly_spaced(items: list[Chunk], count: int) -> list[Chunk]:
    """按等距抽取，避免只取到文档开头几页。"""
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[min(int(index * step), len(items) - 1)] for index in range(count)]


def _select_chunks(chunks: list[Chunk], max_chunks: int, sample: str) -> list[Chunk]:
    pool = [chunk for chunk in chunks if _is_valuable_chunk(chunk.text)]
    if len(pool) < max_chunks:
        logger.info("高质量切片不足（%d<%d），回退到全部切片", len(pool), max_chunks)
        pool = list(chunks)
    if sample == "random":
        return random.Random(20261115).sample(pool, min(max_chunks, len(pool)))
    if sample == "stratified":
        groups: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in pool:
            groups[chunk.source].append(chunk)
        total = len(pool)
        picked: list[Chunk] = []
        remaining = max_chunks
        sources = sorted(groups)
        for index, source in enumerate(sources):
            items = groups[source]
            if index == len(sources) - 1:
                quota = min(remaining, len(items))
            else:
                share = int(max_chunks * len(items) / total)
                quota = min(max(1, share), len(items), remaining)
            picked.extend(_evenly_spaced(items, quota))
            remaining -= quota
            if remaining <= 0:
                break
        logger.info("分层采样：%d 个切片覆盖 %d 个来源", len(picked), len(sources))
        return picked[:max_chunks]
    return pool[:max_chunks]


def build_eval_set(
    kb: KnowledgeBase,
    *,
    llm: Optional[BaseLLM] = None,
    per_chunk: int = 3,
    max_chunks: int = 30,
    sample: str = "head",
    seed_path: Optional[Path] = None,
    legacy_path: Optional[Path] = None,
    include_unanswerable: bool = True,
    out_path: Optional[Path] = None,
) -> dict[str, Any]:
    chunks = _select_chunks(kb.all_chunks(), max(1, max_chunks), sample)
    if not chunks:
        raise RuntimeError("知识库为空，请先执行 python -m app.cli bootstrap")

    rows: list[dict[str, Any]] = []
    rows.extend(_load_rows(seed_path, "seed"))
    legacy_rows = _load_rows(legacy_path, "legacy")
    _enrich_legacy(legacy_rows, chunks)
    rows.extend(legacy_rows)
    if llm is not None:
        rows.extend(_llm_candidates(llm, chunks, per_chunk=per_chunk))
    else:
        rows.extend(heuristic_candidates(chunks, per_chunk=per_chunk))
    if include_unanswerable:
        rows.extend(
            {
                **item,
                "source": "",
                "page": None,
                "evidence_quote": "",
                "requires_calc": False,
            }
            for item in UNANSWERABLE
        )

    rows = [ground_row(row) for row in rows]

    before_filter = len(rows)
    rows = [row for row in rows if not _is_low_value_question(row.get("question", ""))]
    if before_filter != len(rows):
        logger.info("过滤文档结构/公司治理类问题 %d 条", before_filter - len(rows))

    rows = _dedupe(rows)
    rows = _validate_rows(rows, chunks)
    rows = _dedupe(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = f"fin_q{index:03d}"
        row.pop("_generator", None)

    categories: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    for row in rows:
        categories[row.get("category", "fact")] = categories.get(row.get("category", "fact"), 0) + 1
        difficulties[row.get("difficulty", "easy")] = difficulties.get(row.get("difficulty", "easy"), 0) + 1
    stats = {
        "total": len(rows),
        "answerable": sum(1 for row in rows if row.get("answerable", True)),
        "unanswerable": sum(1 for row in rows if not row.get("answerable", True)),
        "with_evidence": sum(1 for row in rows if row.get("evidence_quote")),
        "by_category": dict(sorted(categories.items())),
        "by_difficulty": dict(sorted(difficulties.items())),
        "generator": "llm" if llm is not None else "heuristic",
        "chunks": len(chunks),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats_path = out_path.with_suffix(".stats.json")
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        stats["path"] = str(out_path)
    return {"rows": rows, "stats": stats}
