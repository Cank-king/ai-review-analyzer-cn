import unittest

from analyzer import _select_representative_reviews


class RepresentativeReviewTest(unittest.TestCase):
    def test_removes_exact_and_near_duplicates(self):
        comments = [
            "物流太慢了，等了好几天才发货。",
            "物流太慢了，等了好几天才发货。",
            "物流太慢了，等了好几天才发货，希望尽快优化配送时效。",
            "包装完整，商品使用方便，整体体验不错。",
        ]
        selected = _select_representative_reviews(comments)
        self.assertLessEqual(len(selected), 3)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertIn("希望尽快优化配送时效", selected[0])


if __name__ == "__main__":
    unittest.main()
