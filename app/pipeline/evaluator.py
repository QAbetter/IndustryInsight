"""RAG 效果评测：检索命中率/MRR 扫描 + 真实 LLM 闭卷/RAG 对比。

支持的 QA 字段见 app/pipeline/eval_schema.py。
旧版只含 answer_keywords 的 qa.jsonl 仍然兼容。
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable

from app.core.llm import BaseLLM
from app.pipeline.eval_schema import normalize_qa_row
from app.rag.knowledge import KnowledgeBase

_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_REFUSAL_MARKERS = (
    "无法确定",
    "无法回答",
    "未提供",
    "未披露",
    "不知道",
    "资料不足",
    "未获取",
    "不适用",
    "cannot determine",
    "not available",
    "no information",
)


def _normalize(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        str(text).replace(",", "").replace("，", "").replace("％", "%"),
    ).lower()


def _normalize_number(value: str) -> str:
    try:
        number = float(value.replace(",", ""))
    except ValueError:
        return value.replace(",", "")
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _numbers(text: str) -> set[str]:
    return {_normalize_number(item) for item in _NUMBER_RE.findall(str(text))}


def load_qa(path: Path | str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(normalize_qa_row(json.loads(line)))
    return rows


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def answer_matches(prediction: str, row: dict[str, Any]) -> bool:
    """数值/别名匹配：适合财务数据；开放题可替换为 LLM-as-judge。"""
    prediction = str(prediction or "")
    normalized = _normalize(prediction)
    if not row.get("answerable", True):
        return any(marker in normalized for marker in _REFUSAL_MARKERS)

    aliases = (
        [row.get("gold_answer", "")]
        + list(row.get("answer_aliases") or [])
        + list(row.get("answer_keywords") or [])
    )
    for alias in aliases:
        normalized_alias = _normalize(alias)
        if normalized_alias and normalized_alias in normalized:
            return True

    gold_numbers = _numbers(row.get("gold_answer", ""))
    prediction_numbers = _numbers(prediction)
    if gold_numbers and prediction_numbers:
        return bool(gold_numbers & prediction_numbers)
    return False


def _relevant(row: dict[str, Any], chunk_text: str, chunk_source: str, chunk_page: int | None) -> bool:
    text = _normalize(chunk_text)
    quote = _normalize(row.get("evidence_quote", ""))
    if quote:
        for length in (60, 40, 24):
            prefix = quote[:length]
            if prefix and prefix in text:
                return True

    keywords = (
        list(row.get("answer_keywords") or [])
        + list(row.get("answer_aliases") or [])
        + [row.get("gold_answer", "")]
    )
    for keyword in keywords:
        normalized_keyword = _normalize(keyword)
        if normalized_keyword and normalized_keyword in text:
            return True

    if not quote and not keywords and row.get("source"):
        return chunk_source == row["source"] and (
            row.get("page") is None or chunk_page == row["page"]
        )
    return False


def evaluate_retrieval(
    kb: KnowledgeBase,
    qa: list[dict[str, Any]],
    top_k_values: Iterable[int] = (2, 4, 8),
    thresholds: Iterable[float] = (0.0, 0.18, 0.3),
) -> list[dict[str, Any]]:
    if kb.store.count() == 0:
        raise RuntimeError("知识库为空，请先执行 python -m app.cli bootstrap")
    rows = [normalize_qa_row(row) for row in qa]
    answerable = [row for row in rows if row.get("answerable", True)]
    if not answerable:
        raise RuntimeError("评测集中没有可回答的问题")

    results: list[dict[str, Any]] = []
    for top_k in top_k_values:
        for threshold in thresholds:
            hits = 0
            reciprocal_rank = 0.0
            latencies: list[float] = []
            missed: list[str] = []
            for row in answerable:
                started = time.perf_counter()
                retrieved = kb.search(
                    row["question"],
                    top_k=int(top_k),
                    threshold=float(threshold),
                    company=row.get("company"),
                    report_year=row.get("report_year"),
                )
                latencies.append((time.perf_counter() - started) * 1000)
                rank = None
                for index, item in enumerate(retrieved):
                    if _relevant(row, item.chunk.text, item.chunk.source, item.chunk.page):
                        rank = index + 1
                        break
                if rank:
                    hits += 1
                    reciprocal_rank += 1.0 / rank
                else:
                    missed.append(row["id"])
            total = max(1, len(answerable))
            results.append(
                {
                    "top_k": int(top_k),
                    "threshold": float(threshold),
                    "questions": len(answerable),
                    "skipped_unanswerable": len(rows) - len(answerable),
                    "hits": hits,
                    "hit_rate": round(hits / total, 4),
                    "mrr": round(reciprocal_rank / total, 4),
                    "avg_latency_ms": round(sum(latencies) / total, 2),
                    "missed": missed,
                }
            )
    return results


def evaluate_qa(
    llm: BaseLLM,
    kb: KnowledgeBase,
    qa: list[dict[str, Any]],
    *,
    top_k: int = 4,
    threshold: float = 0.15,
) -> dict[str, Any]:
    rows = [normalize_qa_row(row) for row in qa]
    details: list[dict[str, Any]] = []
    closed_hits = rag_hits = 0
    closed_answerable_hits = rag_answerable_hits = 0
    answerable_total = 0
    refusal_total = refusal_hits = 0

    for row in rows:
        question = row["question"]
        answerable = bool(row.get("answerable", True))
        closed_prompt = [
            {
                "role": "system",
                "content": "你是行业研究助手。只输出答案本身，不要解释；即使不确定也给出最可能的答案，不要以“无法确定”代替作答。",
            },
            {"role": "user", "content": question},
        ]
        closed = llm.chat(closed_prompt, stage="eval_closed", max_tokens=1024).text

        retrieved = kb.search(
            question,
            top_k=top_k,
            threshold=threshold,
            company=row.get("company"),
            report_year=row.get("report_year"),
        )
        context = "\n\n".join(
            f"[{index}] {item.chunk.source} p.{item.chunk.page or '-'}\n{item.chunk.text}"
            for index, item in enumerate(retrieved, start=1)
        )
        rag_prompt = [
            {
                "role": "system",
                "content": (
                    "你是行业研究助手。仅根据给定资料回答；资料不足时回答“无法确定”。"
                    "只输出答案本身，不要解释。"
                ),
            },
            {"role": "user", "content": f"资料：\n{context or '（无）'}\n\n问题：{question}"},
        ]
        rag = llm.chat(rag_prompt, stage="eval_rag", max_tokens=1024).text

        closed_ok = answer_matches(closed, row)
        rag_ok = answer_matches(rag, row)
        closed_hits += int(closed_ok)
        rag_hits += int(rag_ok)
        if answerable:
            answerable_total += 1
            closed_answerable_hits += int(closed_ok)
            rag_answerable_hits += int(rag_ok)
        else:
            refusal_total += 1
            refusal_hits += int(rag_ok)

        details.append(
            {
                "id": row["id"],
                "question": question,
                "category": row.get("category", "fact"),
                "difficulty": row.get("difficulty", "easy"),
                "answerable": answerable,
                "gold_answer": row.get("gold_answer", ""),
                "closed_book": str(closed).strip()[:200],
                "closed_ok": closed_ok,
                "rag": str(rag).strip()[:200],
                "rag_ok": rag_ok,
                "retrieved": len(retrieved),
            }
        )

    total = max(1, len(rows))
    by_category: dict[str, dict[str, Any]] = {}
    for item in details:
        bucket = by_category.setdefault(
            item["category"], {"n": 0, "closed_correct": 0, "rag_correct": 0}
        )
        bucket["n"] += 1
        bucket["closed_correct"] += int(item["closed_ok"])
        bucket["rag_correct"] += int(item["rag_ok"])
    for bucket in by_category.values():
        bucket["closed_accuracy"] = round(bucket["closed_correct"] / bucket["n"], 4)
        bucket["rag_accuracy"] = round(bucket["rag_correct"] / bucket["n"], 4)

    closed_accuracy = closed_hits / total
    rag_accuracy = rag_hits / total
    closed_ci = wilson_interval(closed_hits, len(rows))
    rag_ci = wilson_interval(rag_hits, len(rows))
    return {
        "questions": len(rows),
        "answerable_questions": answerable_total,
        "unanswerable_questions": refusal_total,
        "top_k": top_k,
        "threshold": threshold,
        "closed_book_accuracy": round(closed_accuracy, 4),
        "rag_accuracy": round(rag_accuracy, 4),
        "delta_pp": round((rag_accuracy - closed_accuracy) * 100, 1),
        "closed_book_answerable_accuracy": round(
            closed_answerable_hits / max(1, answerable_total), 4
        ),
        "rag_answerable_accuracy": round(
            rag_answerable_hits / max(1, answerable_total), 4
        ),
        "rag_refusal_accuracy": round(refusal_hits / max(1, refusal_total), 4),
        "closed_book_ci95": [round(closed_ci[0], 4), round(closed_ci[1], 4)],
        "rag_ci95": [round(rag_ci[0], 4), round(rag_ci[1], 4)],
        "by_category": by_category,
        "details": details,
    }
