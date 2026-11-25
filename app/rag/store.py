"""向量库：内置 NumPy 持久化实现 + 可选 Chroma 适配。"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import RAGSettings
from app.rag.models import Chunk

logger = logging.getLogger("insight.store")


class IndexMismatchError(RuntimeError):
    """索引与当前 Embedding 配置不一致。"""


class VectorStore(ABC):
    backend = "base"

    @abstractmethod
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        company: Optional[str] = None,
        report_year: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def all_ids(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    def save(self) -> None:
        return None


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class LocalVectorStore(VectorStore):
    backend = "local"

    def __init__(self, path: Path | str, embedder_name: str, dim: int):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.load_error: Optional[str] = None
        self.load()

    def _files(self) -> tuple[Path, Path, Path]:
        return self.path / "vectors.npy", self.path / "chunks.jsonl", self.path / "meta.json"

    def load(self) -> None:
        vectors_path, chunks_path, meta_path = self._files()
        if not (vectors_path.exists() and chunks_path.exists() and meta_path.exists()):
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embedder") != self.embedder_name:
                raise IndexMismatchError(
                    f"索引由 {meta.get('embedder')} 构建，当前配置为 {self.embedder_name}，请重建索引"
                )
            vectors = np.load(vectors_path)
            chunks = [
                Chunk.from_dict(json.loads(line))
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(chunks) != len(vectors):
                raise IndexMismatchError("索引文件损坏：chunks 与 vectors 数量不一致")
            self.vectors = np.asarray(vectors, dtype=np.float32)
            self.chunks = chunks
            self.dim = int(self.vectors.shape[1]) if self.vectors.size else self.dim
        except IndexMismatchError as exc:
            self.load_error = str(exc)
            logger.warning(self.load_error)
            self.chunks = []
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"索引加载失败：{exc}"
            logger.warning(self.load_error)
            self.chunks = []
            self.vectors = np.zeros((0, self.dim), dtype=np.float32)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        matrix = _normalize(vectors)
        if self.vectors.size == 0:
            self.vectors = matrix
        else:
            self.vectors = np.vstack([self.vectors, matrix]).astype(np.float32)
        self.chunks.extend(chunks)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        company: Optional[str] = None,
        report_year: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.vectors.size == 0:
            return []
        mask = np.ones(len(self.chunks), dtype=bool)
        if company:
            mask &= np.array(
                [chunk.metadata.get("company") == company for chunk in self.chunks],
                dtype=bool,
            )
        if report_year is not None:
            mask &= np.array(
                [chunk.metadata.get("report_year") == int(report_year) for chunk in self.chunks],
                dtype=bool,
            )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        query = _normalize(query_vector)[0]
        subset_scores = self.vectors[indices] @ query
        order = np.argsort(-subset_scores)[: max(1, top_k)]
        return [
            (self.chunks[int(indices[position])], float(subset_scores[position]))
            for position in order
        ]

    def count(self) -> int:
        return len(self.chunks)

    def all_ids(self) -> set[str]:
        return {chunk.id for chunk in self.chunks}

    def save(self) -> None:
        vectors_path, chunks_path, meta_path = self._files()
        np.save(vectors_path, self.vectors)
        with chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        meta_path.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "embedder": self.embedder_name,
                    "dim": int(self.vectors.shape[1]) if self.vectors.size else self.dim,
                    "count": len(self.chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def reset(self) -> None:
        self.chunks = []
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.load_error = None
        self.save()


class ChromaVectorStore(VectorStore):
    backend = "chroma"

    def __init__(self, path: Path | str, collection: str, embedder_name: str, dim: int):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 可选依赖
            raise RuntimeError(
                "未安装 chromadb。请执行 pip install -r requirements-optional.txt，"
                "或把 rag.backend 设为 local。"
            ) from exc
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.dim = dim
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection_name = collection
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        matrix = _normalize(vectors)
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=matrix.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "page": chunk.page if chunk.page is not None else -1,
                    "chunk_index": chunk.chunk_index,
                    "doc_type": chunk.doc_type,
                    "embedder": self.embedder_name,
                    "company": str(chunk.metadata.get("company", "")),
                    "report_year": int(chunk.metadata.get("report_year", 0) or 0),
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        company: Optional[str] = None,
        report_year: Optional[int] = None,
    ) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        query = _normalize(query_vector)[0].tolist()
        clauses: list[dict[str, object]] = []
        if company:
            clauses.append({"company": company})
        if report_year is not None:
            clauses.append({"report_year": int(report_year)})
        where: Optional[dict[str, object]] = None
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}
        payload = self.collection.query(
            query_embeddings=[query],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
            **({"where": where} if where is not None else {}),
        )
        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        results: list[tuple[Chunk, float]] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            page = metadata.get("page", -1)
            results.append(
                (
                    Chunk(
                        id=chunk_id,
                        text=text or "",
                        source=str(metadata.get("source", "")),
                        page=None if int(page) < 0 else int(page),
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        doc_type=str(metadata.get("doc_type", "txt")),
                        metadata=dict(metadata),
                    ),
                    1.0 - float(distance),
                )
            )
        return results

    def count(self) -> int:
        return int(self.collection.count())

    def all_ids(self) -> set[str]:
        try:
            payload = self.collection.get(include=[])
        except Exception:  # noqa: BLE001 - 兼容不同 Chroma 版本
            payload = self.collection.get()
        return set(payload.get("ids", []))

    def all_chunks(self) -> list[Chunk]:
        payload = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for chunk_id, text, metadata in zip(
            payload.get("ids", []), payload.get("documents", []), payload.get("metadatas", [])
        ):
            metadata = metadata or {}
            page = metadata.get("page", -1)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text or "",
                    source=str(metadata.get("source", "")),
                    page=None if int(page) < 0 else int(page),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    doc_type=str(metadata.get("doc_type", "txt")),
                    metadata=dict(metadata),
                )
            )
        return chunks

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


def create_store(settings: RAGSettings, root: Path, embedder_name: str, dim: int) -> VectorStore:
    if settings.backend == "local":
        return LocalVectorStore(root / settings.persist_dir, embedder_name, dim)
    if settings.backend == "chroma":
        return ChromaVectorStore(
            root / settings.chroma_dir, settings.collection, embedder_name, dim
        )
    raise ValueError(f"未知向量库后端：{settings.backend}")
