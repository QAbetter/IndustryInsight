from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.config import Settings
from app.core.tools import ToolRegistry, build_tool_registry, calculate
from app.rag.models import Chunk, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent


class FakeKB:
    def search(self, query, top_k=None, threshold=None):
        chunk = Chunk(
            id="c1",
            text=f"本地研报：{query} 的相关数据是 33.2%",
            source="demo.pdf",
            page=2,
            chunk_index=0,
        )
        return [RetrievedChunk(chunk=chunk, score=0.8, vector_score=0.6, lexical_score=1.0)]


class ToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings.from_file(ROOT / "config" / "settings.yaml")
        cls.registry = build_tool_registry(cls.settings, FakeKB())

    def test_calculator(self):
        payload = json.loads(calculate("(1286.6/136.6)**(1/4)-1"))
        self.assertIn("result", payload)
        with self.assertRaises(Exception):
            calculate("__import__('os').system('dir')")

    def test_search_fixture(self):
        result = self.registry.execute(
            "search_web",
            {"query": "2024 新能源汽车 销量 渗透率", "max_results": 3},
            {"search_web"},
        )
        self.assertTrue(result.ok, result.error)
        payload = json.loads(result.output)
        self.assertGreater(payload["count"], 0)
        self.assertEqual(payload["provider"], "fixture")

    def test_fetch_fixture_and_extract(self):
        search = self.registry.execute(
            "search_web", {"query": "2024 新能源汽车 销量 渗透率"}, {"search_web"}
        )
        url = json.loads(search.output)["results"][0]["url"]
        result = self.registry.execute("fetch_page", {"url": url}, {"fetch_page"})
        self.assertTrue(result.ok, result.error)
        page = json.loads(result.output)
        self.assertIn("1,286.6", page["text"])

    def test_tool_permission(self):
        result = self.registry.execute("search_web", {"query": "x"}, {"kb_search"})
        self.assertFalse(result.ok)
        self.assertIn("未被授权", result.error)

    def test_unknown_tool(self):
        result = self.registry.execute("not_exists", {"query": "x"}, {"not_exists"})
        self.assertFalse(result.ok)
        self.assertIn("不存在", result.error)

    def test_kb_tool_outputs_metadata(self):
        result = self.registry.execute("kb_search", {"query": "比亚迪"}, {"kb_search"})
        self.assertTrue(result.ok)
        payload = json.loads(result.output)
        self.assertEqual(payload["results"][0]["source"], "demo.pdf")
        self.assertEqual(payload["results"][0]["page"], 2)

    def test_registry_schema_description(self):
        text = self.registry.describe({"search_web"})
        self.assertIn("search_web", text)
        self.assertNotIn("kb_search", text)


if __name__ == "__main__":
    unittest.main()
