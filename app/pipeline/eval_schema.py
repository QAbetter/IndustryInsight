"""评测集数据契约：一条 QA 必须能回溯到原文证据。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    id: str
    question: str
    gold_answer: str
    answer_aliases: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    source: str = ""
    page: Optional[int] = None
    category: str = "fact"
    difficulty: str = "easy"
    answerable: bool = True
    requires_calc: bool = False
    company: str = ""
    report_year: Optional[int] = None


def normalize_qa_row(row: dict[str, Any]) -> dict[str, Any]:
    """兼容旧版 answer_keywords 格式，统一成新评测字段。"""
    if "gold_answer" in row:
        payload = dict(row)
        payload["answer_aliases"] = list(payload.get("answer_aliases") or [])
        payload["answer_keywords"] = list(payload.get("answer_keywords") or [])
        payload["evidence_quote"] = payload.get("evidence_quote") or ""
        payload["source"] = payload.get("source") or ""
        payload["page"] = payload.get("page")
        payload["category"] = payload.get("category") or "fact"
        payload["difficulty"] = payload.get("difficulty") or "easy"
        payload["answerable"] = bool(payload.get("answerable", True))
        payload["requires_calc"] = bool(payload.get("requires_calc", False))
        payload["company"] = payload.get("company") or ""
        payload["report_year"] = payload.get("report_year")
        return payload

    keywords = [str(item) for item in row.get("answer_keywords", [])]
    return {
        "id": row.get("id", ""),
        "question": row["question"],
        "gold_answer": keywords[0] if keywords else "",
        "answer_aliases": keywords[1:],
        "answer_keywords": keywords,
        "evidence_quote": row.get("evidence_quote", ""),
        "source": row.get("source", ""),
        "page": row.get("page"),
        "category": row.get("category", "fact"),
        "difficulty": row.get("difficulty", "easy"),
        "answerable": bool(row.get("answerable", True)),
        "requires_calc": bool(row.get("requires_calc", False)),
        "company": row.get("company", ""),
        "report_year": row.get("report_year"),
    }
