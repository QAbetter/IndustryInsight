"""RAG 领域对象。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LoadedDocument:
    text: str
    source: str
    page: Optional[int] = None
    doc_type: str = "txt"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: Optional[int]
    chunk_index: int
    doc_type: str = "txt"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "doc_type": self.doc_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Chunk":
        return cls(
            id=payload["id"],
            text=payload["text"],
            source=payload.get("source", ""),
            page=payload.get("page"),
            chunk_index=int(payload.get("chunk_index", 0)),
            doc_type=payload.get("doc_type", "txt"),
            metadata=payload.get("metadata", {}) or {},
        )


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_score: float = 0.0
    lexical_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "page": self.chunk.page,
            "score": round(self.score, 4),
            "vector_score": round(self.vector_score, 4),
            "lexical_score": round(self.lexical_score, 4),
        }


@dataclass
class IngestStats:
    files: int = 0
    documents: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0
    elapsed_ms: float = 0.0
    backend: str = ""
    embedder: str = ""
    dim: int = 0
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "documents": self.documents,
            "chunks_added": self.chunks_added,
            "chunks_skipped": self.chunks_skipped,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "backend": self.backend,
            "embedder": self.embedder,
            "dim": self.dim,
            "skipped_files": self.skipped_files,
        }
