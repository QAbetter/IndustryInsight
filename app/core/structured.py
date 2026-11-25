"""结构化输出：从自然语言中抽取 JSON、按 Pydantic 校验、Markdown 交付契约校验。"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
_CITATION_RE = re.compile(r"\[E\d+\]")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$", re.M)


def extract_json(text: str) -> Any:
    """尽力从 LLM 输出中提取 JSON，支持代码块、带前后缀说明、多段 JSON。"""
    if not text:
        raise ValueError("空输出，无法提取 JSON")

    fenced = _FENCE_RE.findall(text)
    for block in reversed(fenced):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for match in re.finditer(r"[\[{]", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
            candidates.append(obj)
        except json.JSONDecodeError:
            continue
    if not candidates:
        raise ValueError("输出中未找到合法 JSON")
    # 取信息量最大的候选，避免把示例小对象误当交付物
    return max(candidates, key=lambda obj: len(json.dumps(obj, ensure_ascii=False)))


def validate_json_output(text: str, schema: type[BaseModel]) -> tuple[Optional[BaseModel], list[str]]:
    try:
        payload = extract_json(text)
    except ValueError as exc:
        return None, [str(exc)]
    if not isinstance(payload, dict):
        return None, [f"顶层 JSON 必须是对象，实际为 {type(payload).__name__}"]
    try:
        return schema.model_validate(payload), []
    except ValidationError as exc:
        errors = []
        for item in exc.errors()[:8]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "<root>"
            errors.append(f"{location}: {item.get('msg', 'invalid')}")
        return None, errors


def check_markdown(
    text: str,
    *,
    min_chars: int = 0,
    required_sections: list[str] | None = None,
    min_citations: int = 0,
    require_table: bool = False,
) -> list[str]:
    errors: list[str] = []
    body = text.strip()
    if min_chars and len(body) < min_chars:
        errors.append(f"正文长度 {len(body)} 字，低于要求 {min_chars} 字")
    for section in required_sections or []:
        if section not in body:
            errors.append(f"缺少必需章节：{section}")
    citations = set(_CITATION_RE.findall(body))
    if min_citations and len(citations) < min_citations:
        errors.append(f"证据引用 {len(citations)} 处，少于要求 {min_citations} 处（格式 [E1]）")
    if require_table:
        has_separator = bool(_TABLE_SEP_RE.search(body))
        has_row = bool(_TABLE_ROW_RE.search(body))
        if not (has_separator and has_row):
            errors.append("缺少 Markdown 表格（需包含表头与分隔行）")
    return errors


def correction_prompt(errors: list[str], schema_hint: str = "") -> str:
    lines = [
        "你上一次的输出未通过交付格式校验，问题如下：",
        *[f"- {error}" for error in errors],
        "",
        "请修复上述问题，重新输出**完整**的最终交付物；不要输出解释、道歉或多余前后缀。",
    ]
    if schema_hint:
        lines.append(schema_hint)
    return "\n".join(lines)


def count_citations(text: str) -> int:
    return len(set(_CITATION_RE.findall(text)))
