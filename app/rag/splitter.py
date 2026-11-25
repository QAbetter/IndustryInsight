"""递归字符切片。

优先使用 LangChain 的 RecursiveCharacterTextSplitter（安装 langchain-text-splitters 时），
未安装则使用等价的本地实现，保证项目零重型依赖也能运行。
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("insight.splitter")

try:  # pragma: no cover - 依赖是否安装决定分支
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainSplitter
except Exception:  # noqa: BLE001
    _LangChainSplitter = None

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        separators: Optional[list[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self._impl = None
        if _LangChainSplitter is not None:
            self._impl = _LangChainSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.separators,
                keep_separator=True,
            )

    def split_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        if self._impl is not None:
            chunks = self._impl.split_text(text)
        else:
            chunks = self._split(text, self.separators)
        cleaned = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
        return cleaned

    # ------------------------------------------------------------------ #
    # 本地等价实现
    # ------------------------------------------------------------------ #
    def _split(self, text: str, separators: list[str]) -> list[str]:
        separator = separators[-1]
        remaining: list[str] = []
        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = ""
                remaining = []
                break
            if candidate in text:
                separator = candidate
                remaining = separators[index + 1 :]
                break

        splits = text.split(separator) if separator else list(text)
        good: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                good.append(piece)
            elif remaining:
                good.extend(self._split(piece, remaining))
            else:
                good.extend(self._hard_split(piece))
        return self._merge(good, separator)

    def _hard_split(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            text[start : start + self.chunk_size]
            for start in range(0, len(text), step)
            if text[start : start + self.chunk_size]
        ]

    def _merge(self, splits: list[str], separator: str) -> list[str]:
        separator_len = len(separator)
        docs: list[str] = []
        current: list[str] = []
        total = 0
        for piece in splits:
            piece_len = len(piece)
            if current and total + piece_len + separator_len > self.chunk_size:
                doc = separator.join(current).strip()
                if doc:
                    docs.append(doc)
                while current and (
                    total > self.chunk_overlap
                    or (total + piece_len + separator_len > self.chunk_size and total > 0)
                ):
                    removed = len(current[0]) + (separator_len if len(current) > 1 else 0)
                    total -= removed
                    current.pop(0)
            current.append(piece)
            total += piece_len + (separator_len if len(current) > 1 else 0)
        if current:
            doc = separator.join(current).strip()
            if doc:
                docs.append(doc)
        return docs
