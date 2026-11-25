"""研报来源元数据：从文件名推断公司、报告年份与报告名。

检索时用 company / report_year 过滤候选切片，避免四家年报中同名科目互相干扰。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

SOURCE_CONTEXTS: dict[str, dict[str, Any]] = {
    "estun": {
        "company": "埃斯顿",
        "report_year": 2024,
        "report_name": "埃斯顿2024年年报",
        "prefix": "在埃斯顿2024年年报中，",
    },
    "topstar": {
        "company": "拓斯达",
        "report_year": 2024,
        "report_name": "拓斯达2024年年报",
        "prefix": "在拓斯达2024年年报中，",
    },
    "inovance": {
        "company": "汇川技术",
        "report_year": 2024,
        "report_name": "汇川技术2024年年报",
        "prefix": "在汇川技术2024年年报中，",
    },
    "stepelectric": {
        "company": "新时达",
        "report_year": 2024,
        "report_name": "新时达2024年年报",
        "prefix": "在新时达2024年年报中，",
    },
    "ifr_world_robotics_2025": {
        "company": "IFR",
        "report_year": 2025,
        "report_name": "IFR《World Robotics 2025》执行摘要",
        "prefix": "根据IFR《World Robotics 2025》执行摘要，",
    },
    "ifr_world_robotics_2024": {
        "company": "IFR",
        "report_year": 2024,
        "report_name": "IFR《World Robotics 2024》执行摘要",
        "prefix": "根据IFR《World Robotics 2024》执行摘要，",
    },
}


def infer_source_context(source: str) -> Optional[dict[str, Any]]:
    """从研报文件名推断公司、报告年份和报告名；无法识别时返回 None。"""
    name = Path(str(source or "").replace("\\", "/")).name.lower()
    for key, context in SOURCE_CONTEXTS.items():
        if key in name:
            return dict(context)
    return None


def strip_report_context(question: str) -> str:
    """去掉题干里的“在XX年报中，”前缀，只保留核心问题用于切片排序。

    公司/年份已由元数据过滤承担，前缀会稀释同一份年报内部的切片区分度。
    """
    text = str(question or "").strip()
    for context in SOURCE_CONTEXTS.values():
        prefix = context["prefix"]
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text
