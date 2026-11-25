from __future__ import annotations

import unittest

from app.core.agent import extract_final_answer, parse_action


class AgentProtocolTest(unittest.TestCase):
    def test_parse_action_with_plain_json(self):
        name, args = parse_action(
            'Thought: 需要检索\nAction: search_web\nAction Input: {"query": "新能源"}'
        )
        self.assertEqual(name, "search_web")
        self.assertEqual(args, {"query": "新能源"})

    def test_parse_action_with_code_fence(self):
        name, args = parse_action(
            "Thought: 需要检索\n"
            "Action: search_web\n"
            "Action Input:\n"
            "```json\n"
            '{"query": "2024 新能源汽车", "max_results": 5}\n'
            "```"
        )
        self.assertEqual(name, "search_web")
        self.assertEqual(args, {"query": "2024 新能源汽车", "max_results": 5})

    def test_extract_final_answer(self):
        self.assertEqual(
            extract_final_answer("Thought: 完成\nFinal Answer: {\"ok\": true}"),
            '{"ok": true}',
        )


if __name__ == "__main__":
    unittest.main()
