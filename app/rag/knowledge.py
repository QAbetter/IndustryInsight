"""知识库：解析 -> 切片 -> 向量化 -> 入库 -> Top-K 检索（带相似度阈值）。"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional

from app.config import RAGSettings, Settings
from app.rag.embeddings import create_embedder
from app.rag.loaders import iter_files, load_file
from app.rag.metadata import infer_source_context, strip_report_context
from app.rag.models import Chunk, IngestStats, RetrievedChunk
from app.rag.splitter import RecursiveCharacterTextSplitter
from app.rag.store import IndexMismatchError, create_store

logger = logging.getLogger("insight.kb")


def _chunk_id(source: str, page: Optional[int], index: int, text: str) -> str:
    payload = f"{source}|{page}|{index}|{text[:48]}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z][a-z0-9_\-]{1,}")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUERY_NOISE = ("是多少", "是什么", "什么时候", "有哪些", "怎么样", "为什么", "如何", "哪些", "什么", "是否", "多少")


def _lexical_terms(text: str) -> set[str]:
    lowered = text.lower()
    for phrase in _QUERY_NOISE:
        lowered = lowered.replace(phrase, " ")
    terms: set[str] = set()
    for match in _CJK_RUN_RE.finditer(lowered):
        run = match.group(0)
        terms.update(run[index : index + 2] for index in range(len(run) - 1))
        terms.update(run[index : index + 3] for index in range(len(run) - 2))
    terms.update(_WORD_RE.findall(lowered))
    terms.update(_NUMBER_RE.findall(lowered))
    return terms


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rag: RAGSettings = settings.rag
        self.embedder = create_embedder(self.rag)
        self.store = create_store(
            self.rag, settings.root, self.embedder.name, self.embedder.dim
        )
        self.load_error: Optional[str] = None
        load_error = getattr(self.store, "load_error", None)
        if load_error:
            self.load_error = load_error
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag.chunk_size, chunk_overlap=self.rag.chunk_overlap
        )

    # ------------------------------------------------------------------ #
    def ingest(self, paths: Iterable[Path | str], *, reset: bool = False) -> IngestStats:
        started = time.perf_counter()
        stats = IngestStats(backend=self.store.backend, embedder=self.embedder.name)
        if self.load_error:
            logger.warning("检测到索引不可用（%s），自动重建", self.load_error)
            self.store.reset()
            self.load_error = None
        if reset:
            self.store.reset()

        files = [path.resolve() for path in iter_files(paths)]
        stats.files = len(files)
        if not files:
            stats.elapsed_ms = (time.perf_counter() - started) * 1000
            return stats

        data_raw = self.settings.path("data/raw").resolve()
        new_chunks: list[Chunk] = []
        existing = self.store.all_ids()
        for path in files:
            base_dir = data_raw if data_raw in path.resolve().parents else path.parent
            try:
                documents = load_file(path, base_dir=base_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("解析失败 %s: %s", path.name, exc)
                stats.skipped_files.append(path.name)
                continue
            stats.documents += len(documents)
            for document in documents:
                pieces = self.splitter.split_text(document.text)
                for index, piece in enumerate(pieces):
                    chunk_id = _chunk_id(document.source, document.page, index, piece)
                    if chunk_id in existing:
                        stats.chunks_skipped += 1
                        continue
                    existing.add(chunk_id)
                    context = infer_source_context(document.source) or {}
                    metadata = {"filename": path.name}
                    if context:
                        metadata.update(
                            {
                                "company": context["company"],
                                "report_year": context["report_year"],
                                "report_name": context["report_name"],
                            }
                        )
                    new_chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=piece,
                            source=document.source,
                            page=document.page,
                            chunk_index=index,
                            doc_type=document.doc_type,
                            metadata=metadata,
                        )
                    )

        if new_chunks:
            vectors = self.embedder.embed_documents([chunk.text for chunk in new_chunks])
            self.store.add(new_chunks, vectors)
            self.store.save()
            stats.chunks_added = len(new_chunks)
        if self.store.backend == "local":
            vectors = getattr(self.store, "vectors", None)
            stats.dim = int(vectors.shape[1]) if vectors is not None and vectors.size else self.embedder.dim
        else:
            stats.dim = self.embedder.dim
        stats.elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "知识库入库完成：files=%d chunks=+%d skipped=%d backend=%s",
            stats.files,
            stats.chunks_added,
            stats.chunks_skipped,
            stats.backend,
        )
        return stats

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        company: Optional[str] = None,
        report_year: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        if self.load_error:
            logger.warning("知识库索引不可用：%s", self.load_error)
            return []
        if self.store.count() == 0:
            return []
        # 公司/年份由元数据过滤承担；去掉题干前缀后只用核心问题排序
        search_query = strip_report_context(query) if (company or report_year is not None) else query
        query_vector = self.embedder.embed_query(search_query)
        limit = int(top_k or self.rag.top_k)
        cutoff = float(self.rag.score_threshold if threshold is None else threshold)
        candidate_limit = min(max(limit * 8, 30), 200)
        rows = self.store.search(
            query_vector,
            candidate_limit,
            company=company,
            report_year=report_year,
        )
        query_terms = _lexical_terms(search_query)
        scored: list[RetrievedChunk] = []
        for chunk, vector_score in rows:
            lexical = 0.0
            if query_terms:
                chunk_terms = _lexical_terms(chunk.text)
                lexical = len(query_terms & chunk_terms) / len(query_terms)
            # 混合检索：向量语义 + 字符/词级词法重合，缓解长文档向量被稀释的问题
            score = 0.55 * max(0.0, vector_score) + 0.45 * lexical
            scored.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    lexical_score=lexical,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for item in scored:
            if item.score < cutoff or item.chunk.id in seen:
                continue
            seen.add(item.chunk.id)
            results.append(item)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        sources = set()
        if self.store.backend == "local":
            sources = {chunk.source for chunk in getattr(self.store, "chunks", [])}
        return {
            "backend": self.store.backend,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "chunks": self.store.count(),
            "sources": len(sources),
            "path": str(
                self.settings.path(
                    self.rag.persist_dir if self.store.backend == "local" else self.rag.chroma_dir
                )
            ),
            "load_error": self.load_error,
        }

    def list_sources(self) -> list[dict]:
        if self.store.backend != "local":
            return []
        grouped: dict[str, dict] = {}
        for chunk in getattr(self.store, "chunks", []):
            row = grouped.setdefault(
                chunk.source,
                {"source": chunk.source, "doc_type": chunk.doc_type, "chunks": 0, "pages": set()},
            )
            row["chunks"] += 1
            if chunk.page:
                row["pages"].add(chunk.page)
        return [
            {
                "source": row["source"],
                "doc_type": row["doc_type"],
                "chunks": row["chunks"],
                "pages": sorted(row["pages"]),
            }
            for row in sorted(grouped.values(), key=lambda item: item["source"])
        ]

    def all_chunks(self) -> list[Chunk]:
        if self.store.backend == "local":
            return list(getattr(self.store, "chunks", []))
        getter = getattr(self.store, "all_chunks", None)
        return getter() if callable(getter) else []

    def reset(self) -> None:
        self.store.reset()
        self.load_error = None
