import unittest

import numpy as np
import pandas as pd

from src.data_loader import (CALIFORNIA, SYNTHETIC, DatasetError, load_dataset, make_synthetic,
                             validate_and_clean)
from src.preprocessing import build_preprocessor, validate_inputs
from tests.helpers import make_california_like


def feature_meta(spec=CALIFORNIA, df=None):
    df = df if df is not None else make_california_like()
    return [{"name": f.name, "label": f.label, "min_allowed": f.min_allowed, "max_allowed": f.max_allowed,
             "train_min": float(df[f.name].min()), "train_max": float(df[f.name].max())} for f in spec.features]


class DataValidationTests(unittest.TestCase):
    def test_valid_data_passes_and_reports(self):
        df, rep = validate_and_clean(make_california_like(), CALIFORNIA)
        self.assertEqual(rep.n_rows_used, 600)
        self.assertEqual(set(rep.outliers_per_feature), set(CALIFORNIA.feature_names))
        self.assertLessEqual(rep.target_min, rep.target_median)
        self.assertLessEqual(rep.target_median, rep.target_max)

    def test_missing_column_raises(self):
        with self.assertRaises(DatasetError):
            validate_and_clean(make_california_like().drop(columns=["MedInc"]), CALIFORNIA)

    def test_bad_rows_removed_and_counted(self):
        df = make_california_like(100)
        df.loc[0, "price_usd"] = np.nan
        df.loc[1, "price_usd"] = -5
        df["MedInc"] = df["MedInc"].astype(object)
        df.loc[2, "MedInc"] = "oops"           # becomes NaN, kept (imputed later) but counted
        df = pd.concat([df, df.iloc[[10]]], ignore_index=True)  # exact duplicate
        clean, rep = validate_and_clean(df, CALIFORNIA)
        self.assertEqual(rep.n_missing_target_removed, 1)
        self.assertEqual(rep.n_nonpositive_target_removed, 1)
        self.assertEqual(rep.n_duplicate_rows_removed, 1)
        self.assertEqual(rep.missing_values_per_feature["MedInc"], 1)
        self.assertEqual(len(clean), 100 - 2)

    def test_too_few_rows_raises(self):
        with self.assertRaises(DatasetError):
            validate_and_clean(make_california_like(10), CALIFORNIA)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(DatasetError):
            load_dataset("nope")

    def test_synthetic_is_demo_and_reproducible(self):
        a, b = make_synthetic(seed=3), make_synthetic(seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(list(a.columns), SYNTHETIC.feature_names + ["Price"])
        self.assertTrue(SYNTHETIC.is_demo)
        self.assertFalse(CALIFORNIA.is_demo)


class PreprocessorTests(unittest.TestCase):
    def test_scaler_fitted_on_training_data_only(self):
        df = make_california_like()
        train, test = df.iloc[:400], df.iloc[400:]
        prep = build_preprocessor(CALIFORNIA.feature_names, scale=True).fit(train[CALIFORNIA.feature_names])
        scaler = prep.named_transformers_["num"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, train[CALIFORNIA.feature_names].mean().to_numpy())
        self.assertFalse(np.allclose(scaler.mean_, df[CALIFORNIA.feature_names].mean().to_numpy()))
        out = prep.transform(test[CALIFORNIA.feature_names])
        self.assertEqual(out.shape, (200, 8))

    def test_missing_values_are_imputed_with_training_median(self):
        df = make_california_like(100)[CALIFORNIA.feature_names]
        prep = build_preprocessor(CALIFORNIA.feature_names, scale=False).fit(df)
        probe = df.iloc[[0]].copy()
        probe["MedInc"] = np.nan
        out = prep.transform(probe)
        self.assertFalse(np.isnan(out).any())
        self.assertAlmostEqual(out[0, 0], df["MedInc"].median())

    def test_no_scaling_for_tree_preprocessor(self):
        prep = build_preprocessor(["a"], scale=False)
        names = [n for n, _ in prep.transformers[0][1].steps]
        self.assertEqual(names, ["impute"])


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.meta = feature_meta()
        self.ok = {f["name"]: (f["train_min"] + f["train_max"]) / 2 for f in self.meta}

    def test_valid_input_gives_one_row_frame_in_feature_order(self):
        frame, warns = validate_inputs(self.ok, self.meta)
        self.assertEqual(frame.shape, (1, 8))
        self.assertEqual(list(frame.columns), CALIFORNIA.feature_names)
        self.assertEqual(warns, [])

    def test_missing_none_empty_and_nonnumeric_rejected(self):
        for bad in (None, "", "abc"):
            vals = dict(self.ok, MedInc=bad)
            with self.assertRaises(ValueError):
                validate_inputs(vals, self.meta)
        vals = dict(self.ok)
        del vals["HouseAge"]
        with self.assertRaises(ValueError):
            validate_inputs(vals, self.meta)

    def test_nan_inf_and_hard_bounds_rejected(self):
        for bad in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                validate_inputs(dict(self.ok, MedInc=bad), self.meta)
        with self.assertRaises(ValueError):
            validate_inputs(dict(self.ok, Latitude=123), self.meta)

    def test_out_of_training_range_warns_but_works(self):
        _, warns = validate_inputs(dict(self.ok, Population=1e7), self.meta)
        self.assertEqual(len(warns), 1)
        self.assertIn("extrapolation", warns[0])

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs(dict(self.ok, Bathrooms=2), self.meta)


if __name__ == "__main__":
    unittest.main()
