import unittest
from pathlib import Path

import pandas as pd

from analyzer import (
    analyze_reviews,
    build_issue_priorities,
    calculate_health_score,
    calculate_issue_priority,
    health_level,
    issue_priority_label,
    _report_input,
)
from pdf_report import build_pdf_report


class BusinessScoringTest(unittest.TestCase):
    def test_health_score_range_and_good_data_is_higher(self):
        good = analyze_reviews(pd.DataFrame({"评论内容": ["质量很好，使用方便"] * 8, "评分": [5] * 8}), "评论内容", "评分")
        bad = analyze_reviews(pd.DataFrame({"评论内容": ["物流很慢，包装破损"] * 8, "评分": [1] * 8}), "评论内容", "评分")
        self.assertGreaterEqual(good["health_score"], 0)
        self.assertLessEqual(good["health_score"], 100)
        self.assertGreaterEqual(bad["health_score"], 0)
        self.assertLessEqual(bad["health_score"], 100)
        self.assertGreater(good["health_score"], bad["health_score"])

    def test_priority_range_order_and_labels(self):
        high = {"mention_count": 80, "negative_count": 30, "negative_share": 75, "severity": "高", "total": 100}
        low = {"mention_count": 3, "negative_count": 1, "negative_share": 2, "severity": "低", "total": 100}
        high_score = calculate_issue_priority(high, 40)
        low_score = calculate_issue_priority(low, 40)
        self.assertGreater(high_score, low_score)
        self.assertTrue(0 <= high_score <= 100)
        self.assertTrue(0 <= low_score <= 100)
        self.assertEqual(issue_priority_label(90), "🔴 P0 紧急")
        self.assertEqual(issue_priority_label(70), "🟠 P1 高")
        self.assertEqual(issue_priority_label(40), "🟡 P2 中")
        self.assertEqual(issue_priority_label(10), "🟢 P3 低")
        self.assertEqual(health_level(95), "优秀")
        self.assertEqual(health_level(55), "高风险")

    def test_ai_input_stays_within_forty_reviews(self):
        result = {"issue_evidence": [{"evidence": [f"评论{i}" for i in range(100)]}], "selling_point_evidence": [], "total": 1000, "positive": 900, "negative": 50, "neutral": 50}
        self.assertLessEqual(len(_report_input(result)["代表性评论证据"]), 40)

    def test_demo_regression_and_pdf(self):
        path = Path(__file__).parents[1] / "demo_reviews_1000.csv"
        if not path.exists():
            self.skipTest("demo_reviews_1000.csv 不存在")
        from analyzer import read_reviews_csv
        frame = read_reviews_csv(path)
        result = analyze_reviews(frame, "评论内容", "评分")
        self.assertEqual(result["total"], 1000)
        self.assertTrue(0 <= result["health_score"] <= 100)
        pdf = build_pdf_report(result, {"执行摘要": "测试"}, "规则版降级")
        self.assertGreater(len(pdf), 1000)
        self.assertIn(b"AI", pdf)


if __name__ == "__main__":
    unittest.main()
