from __future__ import annotations

import unittest

from app.core.structured import check_markdown, extract_json, validate_json_output
from app.schemas import EvidenceBundle, ResearchPlan


class StructuredOutputTest(unittest.TestCase):
    def test_extract_fenced_json(self):
        text = '说明如下：\n```json\n{"topic": "A", "value": 1}\n```\n以上。'
        self.assertEqual(extract_json(text), {"topic": "A", "value": 1})

    def test_extract_json_with_prose_and_example(self):
        text = (
            '示例 {"a": 1}\n'
            '最终结果：{"topic": "新能源汽车", "objectives": ["x"], "sub_questions": [], '
            '"report_outline": ["摘要"], "constraints": []}'
        )
        payload = extract_json(text)
        self.assertEqual(payload["topic"], "新能源汽车")

    def test_validate_schema_reports_errors(self):
        model, errors = validate_json_output('{"objectives": ["x"]}', ResearchPlan)
        self.assertIsNone(model)
        self.assertTrue(errors)

    def test_truncated_evidence_json_is_rejected(self):
        # 模拟 max_tokens 截断：外层对象未闭合，只能解析出完整的单条 item。
        # 严格 schema 必须拒绝它，而不是静默退化成空 EvidenceBundle。
        truncated = (
            '{"topic": "新能源汽车", "items": ['
            '{"id": "E1", "claim": "销量 1286.6 万辆", "detail": "中汽协数据",'
            ' "source_type": "web", "url": "https://a.example/1"},'
            '{"id": "E2", "claim": "出口 128.4 万辆", "detail": "同比 6.7%"'
        )
        model, errors = validate_json_output(truncated, EvidenceBundle)
        self.assertIsNone(model)
        self.assertTrue(errors)

    def test_evidence_bundle_rejects_unknown_field_names(self):
        payload = '{"topic": "x", "evidence": [{"claim": "a"}]}'
        model, errors = validate_json_output(payload, EvidenceBundle)
        self.assertIsNone(model)
        self.assertTrue(errors)

    def test_evidence_bundle_accepts_valid_payload(self):
        payload = (
            '{"topic": "x", "items": ['
            '{"claim": "销量 1 万辆", "source_type": "web", "url": "https://a.example/1"}],'
            ' "gaps": []}'
        )
        model, errors = validate_json_output(payload, EvidenceBundle)
        self.assertIsNotNone(model)
        self.assertEqual(errors, [])
        self.assertEqual(len(model.items), 1)

    def test_markdown_contract(self):
        report = (
            "## 核心摘要\n内容 [E1]\n\n"
            "## 市场规模\n| 指标 | 数值 |\n| --- | --- |\n| 销量 | 1 |\n"
        )
        errors = check_markdown(
            report,
            min_chars=10,
            required_sections=["核心摘要", "市场规模", "竞争格局"],
            min_citations=1,
            require_table=True,
        )
        self.assertTrue(any("竞争格局" in error for error in errors))
        self.assertEqual(len([error for error in errors if "表格" in error]), 0)


if __name__ == "__main__":
    unittest.main()
