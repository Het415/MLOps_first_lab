import unittest

from src import metrics


class TestStats(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(metrics.mean([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(metrics.mean([-2, 2]), 0.0)

    def test_std(self):
        self.assertAlmostEqual(metrics.std([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)

    def test_min_max_scale(self):
        out = metrics.min_max_scale([10, 20, 30])
        for got, want in zip(out, [0.0, 0.5, 1.0]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(metrics.min_max_scale([7, 7]), [0.0, 0.0])

    def test_z_score_constant(self):
        self.assertEqual(metrics.z_score([4, 4]), [0.0, 0.0])

    def test_bad_values(self):
        for bad in ([], "abc", [1, "2"], [True, 2]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    metrics.mean(bad)


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.y_true = [1, 0, 1, 1, 0, 0, 1, 0]
        self.y_pred = [1, 0, 0, 1, 1, 0, 1, 0]

    def test_confusion(self):
        self.assertEqual(
            metrics.confusion(self.y_true, self.y_pred),
            {"tp": 3, "fp": 1, "tn": 3, "fn": 1},
        )

    def test_scores(self):
        for fn in (metrics.accuracy, metrics.precision, metrics.recall, metrics.f1):
            with self.subTest(fn=fn.__name__):
                self.assertAlmostEqual(fn(self.y_true, self.y_pred), 0.75)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.accuracy([1, 0], [1])

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            metrics.recall([1, 2], [1, 0])


if __name__ == "__main__":
    unittest.main()
