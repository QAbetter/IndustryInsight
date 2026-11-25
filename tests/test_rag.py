from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.config import Settings
from app.rag.knowledge import KnowledgeBase
from app.rag.metadata import infer_source_context, strip_report_context
from app.rag.models import Chunk
from app.rag.store import LocalVectorStore

ROOT = Path(__file__).resolve().parent.parent
DOC = """# 演示研报

2024 年比亚迪新能源汽车市场份额约 33.2%，宁德时代动力电池全球市占率约 37.9%。
2024 年新能源汽车销量 1,286.6 万辆，同比增长 35.5%，渗透率 40.9%。

## 风险

价格战使行业利润率降至 4.3%，欧盟关税最高约 35.3%。
"""


class RagTest(unittest.TestCase):
    def test_source_context_and_strip_prefix(self):
        context = infer_source_context(r"robotics\estun_2024_annual_report.pdf")
        self.assertEqual(context["company"], "埃斯顿")
        self.assertEqual(context["report_year"], 2024)
        self.assertEqual(
            strip_report_context("在埃斯顿2024年年报中，营业收入是多少？"),
            "营业收入是多少？",
        )

    def test_store_filters_company_and_year(self):
        store = LocalVectorStore(self.root / "filtered", "hashing:8", 8)
        chunks = [
            Chunk(
                id="estun",
                text="埃斯顿研发投入",
                source="estun.pdf",
                page=None,
                chunk_index=0,
                metadata={"company": "埃斯顿", "report_year": 2024},
            ),
            Chunk(
                id="inovance",
                text="汇川研发投入",
                source="inovance.pdf",
                page=None,
                chunk_index=1,
                metadata={"company": "汇川技术", "report_year": 2024},
            ),
            Chunk(
                id="ifr",
                text="IFR 安装量",
                source="ifr.pdf",
                page=None,
                chunk_index=2,
                metadata={"company": "IFR", "report_year": 2025},
            ),
        ]
        vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        store.add(chunks, vectors)
        query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        results = store.search(query, 10, company="埃斯顿", report_year=2024)
        self.assertEqual([chunk.id for chunk, _ in results], ["estun"])

        results = store.search(query, 10, company="IFR", report_year=2024)
        self.assertEqual(results, [])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.doc = self.root / "report.md"
        self.doc.write_text(DOC, encoding="utf-8")
        self.settings = Settings.from_file(ROOT / "config" / "settings.yaml").with_overrides(
            rag={
                "embedding": "hashing",
                "dim": 768,
                "persist_dir": str(self.root / "index"),
                "chunk_size": 180,
                "chunk_overlap": 40,
                "top_k": 3,
                "score_threshold": 0.0,
            }
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_ingest_and_search(self):
        kb = KnowledgeBase(self.settings)
        stats = kb.ingest([self.doc])
        self.assertGreater(stats.chunks_added, 0)
        self.assertEqual(kb.store.count(), stats.chunks_added)
        results = kb.search("比亚迪市场份额是多少", top_k=3)
        self.assertTrue(results)
        self.assertTrue(any("33.2%" in item.chunk.text for item in results))
        self.assertGreater(results[0].lexical_score, 0)

    def test_threshold_filters_all(self):
        kb = KnowledgeBase(self.settings)
        kb.ingest([self.doc])
        self.assertEqual(kb.search("比亚迪", threshold=1.01), [])

    def test_persistence_and_deduplication(self):
        kb = KnowledgeBase(self.settings)
        first = kb.ingest([self.doc])
        reloaded = KnowledgeBase(self.settings)
        self.assertEqual(reloaded.store.count(), first.chunks_added)
        again = reloaded.ingest([self.doc])
        self.assertEqual(again.chunks_added, 0)
        self.assertGreaterEqual(again.chunks_skipped, first.chunks_added)

    def test_source_listing(self):
        kb = KnowledgeBase(self.settings)
        kb.ingest([self.doc])
        sources = kb.list_sources()
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source"], "report.md")


if __name__ == "__main__":
    unittest.main()
