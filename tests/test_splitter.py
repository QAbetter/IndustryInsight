from __future__ import annotations

import unittest

from app.rag.splitter import RecursiveCharacterTextSplitter


class SplitterTest(unittest.TestCase):
    def test_chinese_chunks_respect_size(self):
        text = "。".join(
            [f"第{i}段：2024 年新能源汽车销量持续增长，产业链公司加快出海，价格战与利润率成为市场关注焦点" for i in range(30)]
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
        chunks = splitter.split_text(text)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 210 for chunk in chunks))
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_overlap_present(self):
        text = "A" * 100 + "。" + "B" * 100 + "。" + "C" * 100
        splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=30)
        chunks = splitter.split_text(text)
        self.assertGreaterEqual(len(chunks), 2)

    def test_empty_text(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        self.assertEqual(splitter.split_text(""), [])

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
