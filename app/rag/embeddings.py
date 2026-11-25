"""向量化：GLM / OpenAI Embedding 与离线 Hashing Embedding。"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from app.config import ConfigError, RAGSettings

logger = logging.getLogger("insight.embeddings")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z][a-z0-9_\-]{1,}")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_QUERY_NOISE = (
    "请分析",
    "请介绍",
    "是多少",
    "是什么",
    "什么时候",
    "有哪些",
    "怎么样",
    "为什么",
    "如何",
    "多少",
    "哪些",
    "什么",
    "是否",
    "分别",
    "大约",
    "大概",
    "一下",
    "请问",
)


def _clean_query(text: str) -> str:
    """去掉疑问词等噪声，提升短查询的向量匹配质量。"""
    cleaned = text
    for phrase in _QUERY_NOISE:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[?？!！。，,；;：:、]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or text


class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HashingEmbedder(Embedder):
    """字符 n-gram 哈希向量：离线、确定性、对中文友好。"""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def _features(text: str) -> list[tuple[str, float]]:
        lowered = text.lower()
        features: list[tuple[str, float]] = []
        for match in _CJK_RE.finditer(lowered):
            run = match.group(0)
            features.extend((char, 0.35) for char in run)
            features.extend(
                (run[index : index + 2], 1.0) for index in range(len(run) - 1)
            )
            features.extend(
                (run[index : index + 3], 0.6) for index in range(len(run) - 2)
            )
        features.extend((word, 1.0) for word in _ASCII_RE.findall(lowered))
        features.extend((f"#{number}#", 1.2) for number in _NUMBER_RE.findall(lowered))
        return features

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(text) for text in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([_clean_query(text)])[0]


class OpenAIEmbedder(Embedder):
    """OpenAI 兼容的 Embedding 客户端（OpenAI / GLM 共用）。"""

    def __init__(
        self,
        model: str,
        dim: int = 0,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        name_prefix: str = "openai",
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("未安装 openai，无法使用远程 Embedding") from exc
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ConfigError(
                f"远程 Embedding 需要环境变量 {api_key_env}，"
                f"或在 config/settings.yaml 中改用 rag.embedding: hashing"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.name = f"{name_prefix}:{model}"
        self.dim = dim

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        texts = [text.replace("\n", " ") for text in texts]
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            items = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in items)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array = array / norms
        if not self.dim:
            self.dim = int(array.shape[1])
        return array


class GLMEmbedder(OpenAIEmbedder):
    """智谱 GLM Embedding（OpenAI 兼容接口，默认 embedding-3）。"""

    def __init__(
        self,
        model: str = "embedding-3",
        dim: int = 0,
        *,
        api_key_env: str = "ZHIPU_API_KEY",
        base_url: str | None = "https://open.bigmodel.cn/api/paas/v4/",
    ):
        super().__init__(
            model,
            dim,
            api_key_env=api_key_env,
            base_url=base_url,
            name_prefix="zhipu",
        )


def create_embedder(settings: RAGSettings) -> Embedder:
    if settings.embedding == "hashing":
        return HashingEmbedder(settings.dim)
    if settings.embedding == "openai":
        return OpenAIEmbedder(
            settings.embedding_model or "text-embedding-3-small",
            0,
            api_key_env=settings.embedding_api_key_env or "OPENAI_API_KEY",
            base_url=settings.embedding_base_url or os.getenv("OPENAI_BASE_URL"),
            name_prefix="openai",
        )
    if settings.embedding in {"zhipu", "glm"}:
        model = settings.embedding_model or "embedding-3"
        if model.startswith("text-embedding"):
            logger.warning("rag.embedding=zhipu 不支持 %s，已回退为 embedding-3", model)
            model = "embedding-3"
        return GLMEmbedder(
            model,
            0,
            api_key_env=settings.embedding_api_key_env or "ZHIPU_API_KEY",
            base_url=settings.embedding_base_url
            or "https://open.bigmodel.cn/api/paas/v4/",
        )
    raise ConfigError(f"未知 embedding 后端：{settings.embedding}")
