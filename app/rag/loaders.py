"""文档解析：PDF / Word / Markdown / TXT -> 统一 LoadedDocument。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

from app.rag.models import LoadedDocument

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_pdf(path: Path, source: str) -> list[LoadedDocument]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[LoadedDocument] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - 个别页面损坏时跳过
            text = ""
        text = _clean(text)
        if text:
            documents.append(
                LoadedDocument(
                    text=text,
                    source=source,
                    page=index,
                    doc_type="pdf",
                    metadata={"pages": len(reader.pages)},
                )
            )
    return documents


def _load_docx(path: Path, source: str) -> list[LoadedDocument]:
    from docx import Document

    document = Document(str(path))
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = _clean(paragraph.text)
        if text:
            parts.append(text)
    for table_index, table in enumerate(document.tables, start=1):
        rows: list[str] = []
        for row in table.rows:
            cells = [_clean(cell.text).replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            parts.append(f"[表格 {table_index}]\n" + "\n".join(rows))
    text = _clean("\n\n".join(parts))
    if not text:
        return []
    return [
        LoadedDocument(
            text=text,
            source=source,
            page=None,
            doc_type="docx",
            metadata={"tables": len(document.tables)},
        )
    ]


def _load_text(path: Path, source: str) -> list[LoadedDocument]:
    raw = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if raw is None:
        raise ValueError(f"无法识别文本编码：{path.name}")
    text = _clean(raw)
    if not text:
        return []
    doc_type = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "txt"
    return [LoadedDocument(text=text, source=source, page=None, doc_type=doc_type)]


def load_file(path: Path | str, base_dir: Optional[Path | str] = None) -> list[LoadedDocument]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型：{suffix}")
    source = path.name
    if base_dir:
        try:
            source = str(path.resolve().relative_to(Path(base_dir).resolve()))
        except ValueError:
            source = path.name
    if suffix == ".pdf":
        return _load_pdf(path, source)
    if suffix == ".docx":
        return _load_docx(path, source)
    return _load_text(path, source)


def iter_files(paths: Iterable[Path | str]) -> list[Path]:
    files: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped
