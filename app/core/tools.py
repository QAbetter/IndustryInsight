"""工具注册层：统一描述、权限校验、参数解析与执行计时。"""

from __future__ import annotations

import ast
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import Settings
from app.core.search import PageFetcher, build_search_provider

logger = logging.getLogger("insight.tools")


class ToolExecutionError(RuntimeError):
    pass


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_prompt(self) -> str:
        params = json.dumps(self.parameters.get("properties", {}), ensure_ascii=False)
        return f"- {self.name}: {self.description}\n  参数 JSON Schema: {params}"


@dataclass
class ToolCallResult:
    name: str
    ok: bool
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    args: dict[str, Any] = field(default_factory=dict)

    def observation(self) -> str:
        if self.ok:
            return self.output
        return f"ERROR: 工具 {self.name} 执行失败：{self.error}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self, allowed: Optional[set[str]] = None) -> str:
        tools = [
            tool.to_prompt()
            for name, tool in self._tools.items()
            if allowed is None or name in allowed
        ]
        return "\n".join(tools) if tools else "（当前角色无可用工具）"

    @staticmethod
    def _coerce_args(tool: Tool, raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        text = str(raw).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        required = tool.parameters.get("required", [])
        if len(required) == 1:
            return {required[0]: text}
        raise ToolExecutionError("Action Input 必须是 JSON 对象")

    def execute(
        self, name: str, raw_args: Any, allowed: Optional[set[str]] = None
    ) -> ToolCallResult:
        started = time.perf_counter()
        preview: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {"raw": str(raw_args)[:300]}

        if allowed is not None and name not in allowed:
            return ToolCallResult(
                name=name,
                ok=False,
                error=f"角色未被授权使用工具 {name}（允许：{sorted(allowed)}）",
                duration_ms=(time.perf_counter() - started) * 1000,
                args=preview,
            )
        tool = self._tools.get(name)
        if tool is None:
            return ToolCallResult(
                name=name,
                ok=False,
                error=f"工具不存在（可用：{self.names()}）",
                duration_ms=(time.perf_counter() - started) * 1000,
                args=preview,
            )
        try:
            args = self._coerce_args(tool, raw_args)
            declared = set(tool.parameters.get("properties", {}))
            args = {key: value for key, value in args.items() if key in declared}
            missing = [key for key in tool.parameters.get("required", []) if key not in args]
            if missing:
                raise ToolExecutionError(f"缺少必填参数：{missing}")
            preview = args
            output = tool.func(**args)
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001 - 工具失败转为 Observation 交给 Agent 自愈
            logger.warning("工具执行失败 %s: %s", name, exc)
            return ToolCallResult(
                name=name,
                ok=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - started) * 1000,
                args=preview,
            )
        return ToolCallResult(
            name=name,
            ok=True,
            output=output,
            duration_ms=(time.perf_counter() - started) * 1000,
            args=preview,
        )


# --------------------------------------------------------------------------- #
# 计算器
# --------------------------------------------------------------------------- #

_ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
}
_ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return float(_ALLOWED_CONSTANTS[node.id])
        raise ToolExecutionError(f"不允许的变量：{node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
    ):
        left, right = _eval_node(node.left), _eval_node(node.right)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
            ast.Mod: lambda: left % right,
            ast.FloorDiv: lambda: left // right,
        }
        return operations[type(node.op)]()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _ALLOWED_FUNCTIONS:
            raise ToolExecutionError(f"不允许的函数：{node.func.id}")
        args = [_eval_node(arg) for arg in node.args]
        return float(_ALLOWED_FUNCTIONS[node.func.id](*args))
    raise ToolExecutionError(f"不支持的表达式节点：{type(node).__name__}")


def calculate(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        raise ToolExecutionError("表达式为空")
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
        result = int(result)
    else:
        result = round(result, 6)
    return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 工具构建
# --------------------------------------------------------------------------- #


def build_tool_registry(settings: Settings, kb: Any, fetcher: Optional[PageFetcher] = None) -> ToolRegistry:
    registry = ToolRegistry()
    provider = build_search_provider(settings.search, settings.root)
    fetcher = fetcher or PageFetcher(settings.fetch, settings.root)

    def search_web(query: str, max_results: int | None = None) -> str:
        limit = int(max_results or settings.search.max_results)
        limit = max(1, min(limit, 10))
        results = provider.search(str(query).strip(), limit)
        compact = [
            {
                "title": str(item.get("title", ""))[:120],
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", ""))[:240],
                "published_at": str(item.get("published_at", "") or ""),
            }
            for item in results
        ]
        return json.dumps(
            {
                "query": query,
                "provider": provider.name,
                "count": len(compact),
                "results": compact,
                "note": "若 count=0，请改用更泛化的关键词或调用 kb_search。",
            },
            ensure_ascii=False,
        )

    def fetch_page(url: str) -> str:
        page = fetcher.fetch(str(url).strip())
        return json.dumps(page, ensure_ascii=False)

    def kb_search(
        query: str,
        top_k: int | None = None,
        company: str | None = None,
        report_year: int | None = None,
    ) -> str:
        # 单次 Observation 上限 3000 字：限制条数与单条长度，避免 JSON 被截断
        limit = min(int(top_k) if top_k else settings.rag.top_k, 5)
        filters: dict[str, Any] = {}
        if company:
            filters["company"] = str(company).strip()
        if report_year is not None:
            filters["report_year"] = int(report_year)
        results = kb.search(str(query).strip(), top_k=limit, **filters)
        rows = [
            {
                "chunk_id": item.chunk.id,
                "source": item.chunk.source,
                "page": item.chunk.page,
                "company": item.chunk.metadata.get("company", ""),
                "report_year": item.chunk.metadata.get("report_year"),
                "score": round(item.score, 4),
                "vector_score": round(item.vector_score, 4),
                "lexical_score": round(item.lexical_score, 4),
                "text": item.chunk.text[:360],
            }
            for item in results
        ]
        return json.dumps(
            {
                "query": query,
                "company": filters.get("company"),
                "report_year": filters.get("report_year"),
                "top_k": limit,
                "threshold": settings.rag.score_threshold,
                "count": len(rows),
                "results": rows,
                "note": "若 count=0，说明知识库无相关内容、过滤条件过严或相似度低于阈值。",
            },
            ensure_ascii=False,
        )

    registry.register(
        Tool(
            name="search_web",
            description="联网检索行业信息，返回标题、URL、摘要。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，包含年份与关键实体"},
                    "max_results": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
            func=search_web,
        )
    )
    registry.register(
        Tool(
            name="fetch_page",
            description="读取指定 URL 的网页正文，用于精读关键材料。",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "要读取的网页 URL"}},
                "required": ["url"],
            },
            func=fetch_page,
        )
    )
    registry.register(
        Tool(
            name="kb_search",
            description=(
                "检索本地私有研报知识库（PDF/Word），返回片段、文件名、页码与相似度。"
                "问题涉及具体公司或年份时，传入 company/report_year 过滤，避免跨公司同名科目混检。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "语义检索问题或关键词"},
                    "top_k": {"type": "integer", "description": "返回片段数，默认取配置值"},
                    "company": {
                        "type": "string",
                        "description": "公司名，如 埃斯顿/汇川技术/拓斯达/新时达/IFR；不确定时省略",
                    },
                    "report_year": {
                        "type": "integer",
                        "description": "报告年份，如 2024；不确定时省略",
                    },
                },
                "required": ["query"],
            },
            func=kb_search,
        )
    )
    registry.register(
        Tool(
            name="calculator",
            description="安全计算器，支持 + - * / ** % sqrt/log/round 等，用于校验增速、CAGR、份额。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 (1286.6/136.6)**(1/4)-1",
                    }
                },
                "required": ["expression"],
            },
            func=calculate,
        )
    )
    return registry
