import io
import unittest

import pandas as pd

from analyzer import analyze_reviews, read_reviews_csv


class CsvCompatibilityTest(unittest.TestCase):
    def test_utf8_bom_and_english_aliases(self):
        raw = "review_text,rating,source\n商品质量很好,5,店铺A\n".encode("utf-8-sig")
        frame = read_reviews_csv(io.BytesIO(raw))
        self.assertEqual(list(frame.columns), ["review_text", "rating", "source"])
        result = analyze_reviews(frame, "review_text", "rating")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["positive"], 1)

    def test_gb18030_chinese_csv(self):
        raw = "评价内容,星级\n物流很慢，包装破损,2\n".encode("gb18030")
        frame = read_reviews_csv(io.BytesIO(raw))
        result = analyze_reviews(frame, "评价内容", "星级")
        self.assertEqual(result["negative"], 1)
        self.assertIn("物流问题", [item[0] for item in result["negative_reasons"]])

    def test_empty_and_header_only(self):
        with self.assertRaisesRegex(ValueError, "文件为空"):
            read_reviews_csv(b"")
        frame = read_reviews_csv("评论,评分\n".encode("utf-8"))
        self.assertTrue(frame.empty)

    def test_missing_columns_and_extra_column(self):
        frame = pd.DataFrame({"其他": ["文本"], "备注": ["x"]})
        with self.assertRaises(ValueError):
            analyze_reviews(frame, "评论内容", "评分")
        valid = pd.DataFrame({"评论": ["方便好用"], "score": [5], "无关列": ["保留"]})
        result = analyze_reviews(valid, "评论", "score")
        self.assertEqual(result["positive"], 1)

    def test_null_invalid_and_out_of_range_ratings(self):
        frame = pd.DataFrame({"内容": ["很好", "一般", "不错", "不行"], "stars": [5, None, "abc", 6]})
        result = analyze_reviews(frame, "内容", "stars")
        self.assertEqual(result["positive"], 1)
        self.assertEqual(result["neutral"], 3)


if __name__ == "__main__":
    unittest.main()
