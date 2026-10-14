import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import r2_interpretation, regression_metrics
from src.predict import ModelNotAvailableError, load_predictor, load_test_predictions
from src.train import run_training
from tests.helpers import make_california_like, write_csv


class TrainAndPredictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.csv = write_csv(root / "data.csv", make_california_like())
        cls.models = root / "models"
        cls.metrics = run_training("california", cls.csv, cls.models, seed=5)
        cls.predictor = load_predictor(cls.models)
        cls.base = {f["name"]: f["train_median"] for f in cls.predictor.features}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_models_compared_on_same_split(self):
        names = set(self.metrics["models"])
        self.assertTrue({"Linear Regression", "Random Forest"} <= names)
        n_tests = {r["test"]["n"] for r in self.metrics["models"].values()}
        self.assertEqual(n_tests, {self.metrics["split"]["n_test"]})

    def test_selection_by_cv_rmse_and_baseline_excluded(self):
        real = {n: r for n, r in self.metrics["models"].items() if "baseline" not in n}
        best = min(real, key=lambda n: real[n]["cv"]["rmse_mean"])
        self.assertEqual(self.metrics["selection"]["selected_model"], best)

    def test_test_metrics_match_saved_predictions(self):
        preds = load_test_predictions(self.models)
        m = regression_metrics(preds["actual"], preds["predicted"])
        saved = self.metrics["models"][self.predictor.model_name]["test"]
        for k in ("mae", "rmse", "r2"):
            self.assertAlmostEqual(m[k], saved[k], places=6)

    def test_model_beats_mean_baseline_on_learnable_fixture(self):
        base = self.metrics["models"]["Mean baseline (reference)"]["test"]
        best = self.metrics["models"][self.predictor.model_name]["test"]
        self.assertLess(best["rmse"], base["rmse"])
        self.assertAlmostEqual(base["r2"], 0.0, delta=0.1)

    def test_prediction_is_finite_positive_float(self):
        est = self.predictor.predict(self.base)
        self.assertIsInstance(est.price, float)
        self.assertTrue(np.isfinite(est.price) and est.price > 0)

    def test_higher_income_gives_higher_estimate(self):
        lo = self.predictor.predict(dict(self.base, MedInc=2.0)).price
        hi = self.predictor.predict(dict(self.base, MedInc=9.0)).price
        self.assertGreater(hi, lo)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(self.base, MedInc=None))
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(self.base, HouseAge=-3))
        with self.assertRaises(ValueError):
            self.predictor.predict({})

    def test_feature_analysis_is_separate_and_complete(self):
        names = [f["name"] for f in self.predictor.features]
        self.assertEqual(set(self.metrics["linear_analysis"]["coefficients"]), set(names))
        self.assertEqual(set(self.metrics["importance_analysis"]["rmse_increase_mean"]), set(names))
        # on this fixture income dominates price
        imp = self.metrics["importance_analysis"]["rmse_increase_mean"]
        self.assertEqual(max(imp, key=imp.get), "MedInc")

    def test_missing_artifacts_friendly_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ModelNotAvailableError):
                load_predictor(d)

    def test_tampered_model_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "metrics.json").write_text((self.models / "metrics.json").read_text())
            (d / "house_model.joblib").write_bytes(b"garbage")
            with self.assertRaises(ModelNotAvailableError):
                load_predictor(d)

    def test_incompatible_sklearn_version_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            m = json.loads((self.models / "metrics.json").read_text())
            m["environment"]["scikit_learn"] = "0.1.0"
            (d / "metrics.json").write_text(json.dumps(m))
            (d / "house_model.joblib").write_bytes((self.models / "house_model.joblib").read_bytes())
            with self.assertRaises(ModelNotAvailableError):
                load_predictor(d)


class DemoFallbackTests(unittest.TestCase):
    def test_synthetic_fallback_trains_and_is_flagged_demo(self):
        with tempfile.TemporaryDirectory() as d:
            m = run_training("synthetic", None, Path(d) / "m", seed=1)
            self.assertTrue(m["dataset"]["is_demo"])
            self.assertEqual([f["name"] for f in m["features"]], ["Square_Footage", "Bedrooms", "Age_Years"])
            p = load_predictor(Path(d) / "m")
            self.assertGreater(p.predict({"Square_Footage": 2000, "Bedrooms": 3, "Age_Years": 10}).price, 0)


class MetricsTests(unittest.TestCase):
    def test_regression_metrics_known_values(self):
        m = regression_metrics([1, 2, 3], [1, 2, 5])
        self.assertAlmostEqual(m["mae"], 2 / 3)
        self.assertAlmostEqual(m["rmse"], np.sqrt(4 / 3))
        self.assertLess(m["r2"], 1)

    def test_r2_text_honest_for_negative_and_weak(self):
        self.assertIn("worse", r2_interpretation(-0.2))
        self.assertIn("Weak", r2_interpretation(0.1))


if __name__ == "__main__":
    unittest.main()
