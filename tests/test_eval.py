from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.pipeline.eval_schema import normalize_qa_row
from app.pipeline.evaluator import answer_matches, wilson_interval
from app.pipeline.qa_builder import build_eval_set, heuristic_candidates
from app.rag.models import Chunk


class EvaluationTest(unittest.TestCase):
    def test_numeric_and_range_matching(self):
        row = normalize_qa_row(
            {
                "id": "q1",
                "question": "销量是多少？",
                "gold_answer": "1,286.6 万辆",
                "answer_aliases": ["1286.6"],
            }
        )
        self.assertTrue(answer_matches("约 1286.6 万辆", row))
        self.assertFalse(answer_matches("无法确定", row))

        range_row = normalize_qa_row(
            {"id": "q2", "question": "预测？", "gold_answer": "1,550-1,600 万辆"}
        )
        self.assertTrue(answer_matches("1,600 万辆", range_row))
        self.assertFalse(answer_matches("1,500 万辆", range_row))

    def test_unanswerable_refusal(self):
        row = normalize_qa_row(
            {
                "id": "q3",
                "question": "未披露的数据？",
                "gold_answer": "无法确定",
                "answerable": False,
            }
        )
        self.assertTrue(answer_matches("资料未披露，无法确定。", row))
        self.assertFalse(answer_matches("大约 35%", row))

    def test_legacy_row_normalization(self):
        row = normalize_qa_row(
            {"id": "q4", "question": "比亚迪份额？", "answer_keywords": ["33.2%"]}
        )
        self.assertEqual(row["gold_answer"], "33.2%")
        self.assertTrue(row["answerable"])
        self.assertEqual(row["category"], "fact")

    def test_wilson_interval(self):
        low, high = wilson_interval(45, 53)
        self.assertLess(low, 45 / 53)
        self.assertGreater(high, 45 / 53)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_heuristic_builder_and_full_set(self):
        chunk = Chunk(
            id="c1",
            text=(
                "2024 年中国新能源汽车销量达到 1,286.6 万辆，同比增长 35.5%。"
                "比亚迪以 33.2% 的份额保持第一，特斯拉中国份额约 6.0%。"
            ),
            source="demo.pdf",
            page=1,
            chunk_index=0,
        )
        candidates = heuristic_candidates([chunk], per_chunk=3)
        self.assertTrue(candidates)
        self.assertTrue(all(row["evidence_quote"] for row in candidates))

        class FakeKB:
            def all_chunks(self):
                return [chunk]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "qa.jsonl"
            result = build_eval_set(
                FakeKB(),
                seed_path=None,
                legacy_path=None,
                include_unanswerable=True,
                out_path=out,
            )
            self.assertGreater(result["stats"]["total"], 0)
            self.assertEqual(result["stats"]["unanswerable"], 8)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["id"] for row in rows))
            self.assertTrue(any(row["category"] == "unanswerable" for row in rows))


if __name__ == "__main__":
    unittest.main()
