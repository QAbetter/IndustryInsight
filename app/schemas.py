"""Agent 交付物的数据结构（Pydantic），同时作为结构化输出的校验契约。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _StrictModel(BaseModel):
    """证据类交付物：拒绝未知字段，避免模型字段错位后被静默忽略。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SubQuestion(_Model):
    id: str = ""
    question: str
    search_queries: list[str] = Field(default_factory=list)
    data_points: list[str] = Field(default_factory=list)


class ResearchPlan(_Model):
    topic: str
    objectives: list[str] = Field(default_factory=list)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    report_outline: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class EvidenceItem(_StrictModel):
    id: str = ""
    claim: str
    detail: str = ""
    source_type: Literal["web", "kb"] = "web"
    title: str = ""
    url: str = ""
    source: str = ""
    page: int | None = None
    published_at: str = ""
    confidence: float = 0.6
    query: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class EvidenceBundle(_StrictModel):
    topic: str = ""
    items: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    coverage_notes: str = ""
    queries: list[str] = Field(default_factory=list)

    def dedupe_key(self, item: EvidenceItem) -> str:
        locator = (item.url or "").strip().lower() or f"{item.source}:{item.page}"
        claim = re.sub(r"\s+", "", item.claim)[:60]
        return f"{locator}|{claim}"

    def merge(self, *others: "EvidenceBundle") -> "EvidenceBundle":
        items: list[EvidenceItem] = []
        seen: set[str] = set()
        gaps: list[str] = []
        queries: list[str] = []
        notes: list[str] = []
        for bundle in (self, *others):
            for item in bundle.items:
                key = self.dedupe_key(item)
                if key not in seen:
                    seen.add(key)
                    items.append(item)
            for gap in bundle.gaps:
                if gap and gap not in gaps:
                    gaps.append(gap)
            for query in bundle.queries:
                if query and query not in queries:
                    queries.append(query)
            if bundle.coverage_notes:
                notes.append(bundle.coverage_notes)
        merged = EvidenceBundle(
            topic=self.topic or next((b.topic for b in others if b.topic), ""),
            items=items,
            gaps=gaps,
            queries=queries,
            coverage_notes="；".join(dict.fromkeys(notes)),
        )
        merged.reindex()
        return merged

    def reindex(self) -> None:
        for index, item in enumerate(self.items, start=1):
            item.id = f"E{index}"

    def compact(self, max_items: int = 24, detail_chars: int = 500) -> list[dict]:
        rows = []
        for item in self.items[:max_items]:
            rows.append(
                {
                    "id": item.id,
                    "claim": item.claim,
                    "detail": (item.detail or "")[:detail_chars],
                    "source_type": item.source_type,
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "page": item.page,
                    "confidence": round(item.confidence, 2),
                }
            )
        return rows


class Metric(_Model):
    name: str
    value: str
    year: str = ""
    note: str = ""
    source_ids: list[str] = Field(default_factory=list)


class MarketInfo(_Model):
    size: str = ""
    growth_rate: str = ""
    cagr: str = ""
    penetration: str = ""
    forecast: str = ""
    source_ids: list[str] = Field(default_factory=list)


class Competitor(_Model):
    name: str
    share: str = ""
    positioning: str = ""
    source_ids: list[str] = Field(default_factory=list)


class Trend(_Model):
    title: str
    detail: str = ""
    source_ids: list[str] = Field(default_factory=list)


class Risk(_Model):
    title: str
    detail: str = ""
    severity: Literal["高", "中", "低"] = "中"
    source_ids: list[str] = Field(default_factory=list)


class AnalysisReport(_Model):
    topic: str
    executive_summary: str = ""
    market: MarketInfo = Field(default_factory=MarketInfo)
    key_metrics: list[Metric] = Field(default_factory=list)
    competitive_landscape: list[Competitor] = Field(default_factory=list)
    value_chain: list[str] = Field(default_factory=list)
    trends: list[Trend] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class CoverageDecision(_Model):
    sufficient: bool = True
    gaps: list[str] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)
    reason: str = ""


SCHEMA_MAP: dict[str, type[BaseModel]] = {
    "ResearchPlan": ResearchPlan,
    "EvidenceBundle": EvidenceBundle,
    "AnalysisReport": AnalysisReport,
    "CoverageDecision": CoverageDecision,
}
