from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.pipeline.runner import ResearchPipeline
from app.rag.knowledge import KnowledgeBase
from app.sample_data import ensure_demo_corpus

ROOT = Path(__file__).resolve().parent.parent
DEMO_FILES = ("nev_2024_industry_report.pdf", "nev_2024_competition.docx", "nev_policy_risk.md")


class PipelineIntegrationTest(unittest.TestCase):
    def test_offline_pipeline_end_to_end(self):
        ensure_demo_corpus(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # 只使用演示语料，避免受 data/raw 下真实研报的影响
            demo_raw = tmp_path / "raw"
            demo_raw.mkdir()
            for name in DEMO_FILES:
                shutil.copy2(ROOT / "data" / "raw" / name, demo_raw / name)
            settings = Settings.from_file(ROOT / "config" / "settings.yaml").with_overrides(
                llm={"provider": "mock", "model": ""},
                rag={
                    "embedding": "hashing",
                    "dim": 768,
                    "persist_dir": str(tmp_path / "index"),
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "top_k": 5,
                    "score_threshold": 0.15,
                },
                search={"provider": "fixture"},
                fetch={"offline": True},
                pipeline={
                    "output_dir": str(tmp_path / "runs"),
                    "target_chars": 6000,
                    "max_workers": 2,
                    "parallel": True,
                },
            )
            KnowledgeBase(settings).ingest([demo_raw], reset=True)
            pipeline = ResearchPipeline(settings)
            result = pipeline.run("中国新能源汽车行业（2024）")

            self.assertTrue(result.ok, result.errors)
            self.assertGreater(len(result.report), 4000)
            self.assertGreaterEqual(result.report.count("[E"), 6)
            self.assertIn("## 核心摘要", result.report)
            self.assertIn("## 参考资料", result.report)
            self.assertIsNotNone(result.plan)
            self.assertIsNotNone(result.analysis)
            self.assertIsNotNone(result.evidence)
            self.assertGreaterEqual(len(result.evidence["items"]), 6)
            self.assertTrue(result.analysis["market"]["source_ids"])
            self.assertGreater(result.metrics["tokens"]["total"], 0)
            self.assertIn("researcher", result.metrics["by_agent"])
            self.assertIn("writer", result.metrics["by_agent"])
            self.assertTrue(Path(result.artifacts["report"]).exists())
            self.assertTrue(Path(result.artifacts["report_html"]).exists())
            self.assertTrue((result.run_dir / "evidence.json").exists())
            events = (result.run_dir / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("coverage_check", events)
            self.assertIn("tool_call", events)


if __name__ == "__main__":
    unittest.main()
