"""检索与网页读取：Tavily / DuckDuckGo / 离线 fixtures 三种搜索源，统一网页正文抽取。"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from app.config import FetchSettings, SearchSettings

logger = logging.getLogger("insight.search")


class SearchError(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


def _bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.lower())
    grams: set[str] = set()
    for token in normalized.split():
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                grams.add(token)
            grams.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            grams.add(token)
    return grams


class SearchProvider(ABC):
    name = "base"

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        raise NotImplementedError


class FixtureSearch(SearchProvider):
    """离线检索：按关键词命中 + 字符 bigram 重合度，对预置结果排序。"""

    name = "fixture"

    def __init__(self, fixture_dir: Path | str):
        self.fixture_dir = Path(fixture_dir)
        self.entries: list[dict[str, Any]] = []
        if self.fixture_dir.exists():
            for path in sorted(self.fixture_dir.glob("*.json")):
                try:
                    self.entries.append(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    logger.warning("fixtures 文件解析失败：%s", path)

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        query_lower = query.lower()
        query_grams = _bigrams(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self.entries:
            keywords = [str(item) for item in entry.get("keywords", [])]
            keyword_hits = sum(
                1 for keyword in keywords if keyword.lower() in query_lower or query_lower in keyword.lower()
            )
            if keyword_hits == 0:
                continue
            for result in entry.get("results", []):
                blob = " ".join(
                    [
                        str(result.get("title", "")),
                        str(result.get("snippet", "")),
                        " ".join(keywords),
                        str(entry.get("name", "")),
                    ]
                )
                blob_grams = _bigrams(blob)
                overlap = len(query_grams & blob_grams) / max(1, len(query_grams))
                score = keyword_hits * 1.5 + overlap * 4.0
                if score >= 0.75:
                    scored.append((score, result))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for score, result in scored:
            url = str(result.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            item = dict(result)
            item["score"] = round(score, 3)
            results.append(item)
            if len(results) >= max_results:
                break
        return results


class TavilySearch(SearchProvider):
    name = "tavily"

    def __init__(self, api_key_env: str, timeout: int = 20):
        self.api_key_env = api_key_env
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise SearchError(f"缺少 {self.api_key_env}，无法使用 Tavily")
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": (item.get("content") or "")[:800],
                    "published_at": item.get("published_date", "") or "",
                    "score": item.get("score", 0),
                }
            )
        return results[:max_results]


class DuckDuckGoSearch(SearchProvider):
    name = "duckduckgo"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                )
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[dict[str, Any]] = []
        for block in soup.select("div.result"):
            link = block.select_one("a.result__a")
            snippet = block.select_one(".result__snippet")
            if not link:
                continue
            href = link.get("href", "")
            if "uddg=" in href:
                href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
            if not href.startswith("http"):
                continue
            results.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": href,
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                    "published_at": "",
                }
            )
            if len(results) >= max_results:
                break
        return results


class FallbackSearch(SearchProvider):
    def __init__(self, primary: SearchProvider, fallback: SearchProvider):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}->{fallback.name}"

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        try:
            results = self.primary.search(query, max_results)
            if results:
                return results
            logger.info("在线检索无结果，回退离线 fixtures：%s", query)
        except Exception as exc:  # noqa: BLE001 - 联网失败时保证演示可继续
            logger.warning("在线检索失败（%s），回退离线 fixtures：%s", exc, query)
        return self.fallback.search(query, max_results)


def build_search_provider(settings: SearchSettings, root: Path) -> SearchProvider:
    fixture = FixtureSearch(root / settings.fixture_dir)
    name = settings.provider.lower()
    if name == "fixture":
        return fixture
    if name == "tavily":
        return FallbackSearch(
            TavilySearch(settings.tavily_api_key_env, settings.timeout), fixture
        )
    if name == "duckduckgo":
        return FallbackSearch(DuckDuckGoSearch(settings.timeout), fixture)
    if name == "auto":
        if os.getenv(settings.tavily_api_key_env, "").strip():
            return FallbackSearch(
                TavilySearch(settings.tavily_api_key_env, settings.timeout), fixture
            )
        return FallbackSearch(DuckDuckGoSearch(settings.timeout), fixture)
    raise SearchError(f"未知 search.provider: {settings.provider}")


def extract_main_text(html: str, max_chars: int = 6000) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(
        ["script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form", "button"]
    ):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[str] = []
    seen: set[str] = set()
    for element in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if len(text) < 12 or text in seen:
            continue
        seen.add(text)
        blocks.append(text)
    text = "\n".join(blocks)
    if len(text) < 80:
        text = re.sub(r"\n{3,}", "\n\n", root.get_text("\n", strip=True))
    return title, text[:max_chars]


class PageFetcher:
    def __init__(self, settings: FetchSettings, root: Path):
        self.settings = settings
        self.fixture_dir = root / settings.fixture_dir
        self.manifest: dict[str, str] = {}
        manifest_path = self.fixture_dir / "index.json"
        if manifest_path.exists():
            try:
                self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("页面 fixtures manifest 解析失败：%s", manifest_path)

    def _fixture_page(self, url: str) -> dict[str, Any]:
        filename = self.manifest.get(url)
        if not filename:
            raise FetchError(f"离线模式未找到页面映射：{url}")
        path = self.fixture_dir / filename
        if not path.exists():
            raise FetchError(f"离线页面文件不存在：{path.name}")
        title, text = extract_main_text(path.read_text(encoding="utf-8"), self.settings.max_chars)
        return {"url": url, "title": title, "text": text, "provider": "fixture"}

    def fetch(self, url: str) -> dict[str, Any]:
        if self.settings.offline:
            return self._fixture_page(url)
        try:
            response = requests.get(
                url,
                timeout=self.settings.timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding or response.encoding
            title, text = extract_main_text(response.text, self.settings.max_chars)
            if len(text) < 80:
                raise FetchError("正文抽取结果过短，页面可能由 JS 渲染")
            return {"url": url, "title": title, "text": text, "provider": "http"}
        except Exception as exc:  # noqa: BLE001
            if url in self.manifest:
                logger.warning("在线读取失败，回退离线页面：%s", url)
                return self._fixture_page(url)
            raise FetchError(f"页面读取失败：{exc}") from exc
