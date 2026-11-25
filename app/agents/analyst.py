"""分析师 Agent：对多源证据做一致性清洗后输出结构化分析。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import TaskSpec
from app.core.agent import Agent
from app.core.types import TaskResult

logger = logging.getLogger("insight.analyst")


class AnalystAgent(Agent):
    def run(self, task: TaskSpec, deps: dict[str, TaskResult], inputs: dict[str, Any]) -> TaskResult:
        result = super().run(task, deps, inputs)
        parsed = dict(result.parsed or {})
        valid_ids: set[str] = set()
        for dep in deps.values():
            if dep.kind in {"web_research", "kb_research", "merge_evidence"}:
                for item in (dep.parsed or {}).get("items", []):
                    item_id = item.get("id")
                    if item_id:
                        valid_ids.add(item_id)

        dropped: list[str] = []

        def clean(ids: Any) -> list[str]:
            if not isinstance(ids, list):
                return []
            kept = []
            for item_id in ids:
                if item_id in valid_ids and item_id not in kept:
                    kept.append(item_id)
                elif item_id not in valid_ids:
                    dropped.append(str(item_id))
            return kept

        market = parsed.get("market") or {}
        if isinstance(market, dict):
            market["source_ids"] = clean(market.get("source_ids", []))
            parsed["market"] = market
        for key in ("key_metrics", "competitive_landscape", "trends", "risks"):
            for row in parsed.get(key) or []:
                if isinstance(row, dict):
                    row["source_ids"] = clean(row.get("source_ids", []))
        parsed["evidence_ids"] = clean(parsed.get("evidence_ids", []))
        used = set(parsed["evidence_ids"])
        for key in ("key_metrics", "competitive_landscape", "trends", "risks"):
            for row in parsed.get(key) or []:
                if isinstance(row, dict):
                    used.update(row.get("source_ids", []))
        if market.get("source_ids"):
            used.update(market["source_ids"])
        parsed["evidence_ids"] = sorted(
            used, key=lambda value: int(re.sub(r"\D", "", str(value)) or 0)
        )
        if dropped:
            logger.info("分析师输出中移除了 %d 个不存在的证据引用", len(set(dropped)))
        result.parsed = parsed
        result.text = json.dumps(parsed, ensure_ascii=False, indent=2)
        return result
