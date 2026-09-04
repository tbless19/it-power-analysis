"""Unit tests for Ganglia parameter-correlation helpers."""
import unittest

import numpy as np

from rl_tune.correlate_params import (
    features_from_slice,
    mape,
    mean_field_power,
    partial_r,
    pearson,
    spearman,
)


class TestCorrHelpers(unittest.TestCase):
    def test_pearson_perfect_and_anticorrelated(self):
        x = np.arange(20, dtype=float)
        self.assertAlmostEqual(pearson(x, 2 * x + 1), 1.0, places=6)
        self.assertAlmostEqual(pearson(x, -x), -1.0, places=6)
        self.assertIsNone(pearson(np.ones(10), x[:10]))

    def test_spearman_monotonic(self):
        x = np.linspace(0, 1, 30)
        y = x ** 3
        self.assertGreater(spearman(x, y), 0.99)

    def test_partial_r_removes_confounder(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=200)
        x = z + rng.normal(size=200) * 0.05
        y = z + rng.normal(size=200) * 0.05
        self.assertGreater(pearson(x, y), 0.9)
        self.assertLess(abs(partial_r(y, x, z.reshape(-1, 1))), 0.2)

    def test_mean_field_scales_with_util(self):
        n = 8
        feats = {
            "n_eff_training": np.full(n, 100.0),
            "n_eff_fine_tuning": np.zeros(n),
            "n_eff_inference": np.zeros(n),
            "n_eff_idle": np.full(n, 50.0),
        }
        p0 = mean_field_power(feats, 0.5, 0.0, 0.0, 0.1, p_max_w=300.0)
        p1 = mean_field_power(feats, 1.0, 0.0, 0.0, 0.1, p_max_w=300.0)
        self.assertTrue(np.allclose(p0, 100 * 0.5 * 0.3 + 50 * 0.1 * 0.3))
        self.assertTrue(np.all(p1 > p0))

    def test_mape(self):
        meas = np.array([100.0, 200.0, 0.0])
        pred = np.array([110.0, 180.0, 1.0])
        self.assertAlmostEqual(mape(pred, meas), (0.1 + 0.1) / 2 * 100.0, places=6)

    def test_analyze_ranks_fine_tuning_against_ganglia_slice(self):
        from rl_tune.correlate_params import analyze

        report = analyze()
        by = {r["parameter"]: r for r in report["parameters"]}
        self.assertGreater(by["fine_tuning.u_plateau"]["pearson_r_vs_ganglia"], 0.7)
        self.assertGreater(by["fine_tuning.u_plateau"]["partial_r_vs_ganglia"], 0.7)
        self.assertIn(
            "fine_tuning.u_plateau",
            report["rl_search_recommendation"]["search_first"],
        )
        self.assertIn("rho", report["rl_search_recommendation"]["defer"])
        self.assertIn(
            "training.u_plateau",
            report["rl_search_recommendation"]["confounders"],
        )

    def test_features_from_slice_shapes(self):
        data = {
            "measured": {"kw": [10.0, 20.0, 30.0, 40.0]},
            "modeled": {
                "fleet_kw": [11.0, 19.0, 29.0, 41.0],
                "alloc_gpus": [100, 200, 300, 400],
                "active_nodes": {
                    "training": [10, 10, 20, 5],
                    "fine_tuning": [5, 15, 10, 20],
                    "inference": [20, 10, 5, 10],
                },
                "active_jobs": {
                    "training": [1, 1, 2, 1],
                    "fine_tuning": [2, 3, 2, 4],
                    "inference": [8, 4, 2, 3],
                },
                "by_stage": {
                    "training": [1, 1, 2, 0.5],
                    "fine_tuning": [1, 2, 1, 3],
                    "inference": [2, 1, 0.5, 1],
                },
            },
            "meta": {"inventory": {"V100": 1000}, "hw_type": "V100"},
        }
        feats = features_from_slice(data)
        self.assertEqual(len(feats["ganglia_kw"]), 4)
        self.assertTrue(np.allclose(feats["n_req_training"], [40, 40, 80, 20]))
        self.assertTrue(np.all(feats["n_eff_idle"] >= 0))


if __name__ == "__main__":
    unittest.main()
