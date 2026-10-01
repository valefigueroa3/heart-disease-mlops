"""Pruebas pequeñas de las reglas de preparación de datos."""

import unittest

import numpy as np
import pandas as pd

from app.modeling import build_preprocessor, prepare_features_and_target


class PrepareDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                "Age": [40, 50, 60, 70],
                "Sex": ["M", "F", "M", "F"],
                "RestingBP": [120, 0, 130, 140],
                "Cholesterol": [0, 220, 200, 0],
                "HeartDisease": [0, 1, 0, 1],
            }
        )

    def test_zero_measurements_become_missing_and_target_is_separate(self) -> None:
        X, y = prepare_features_and_target(self.data)

        self.assertNotIn("HeartDisease", X.columns)
        self.assertEqual(y.tolist(), [0, 1, 0, 1])
        self.assertTrue(pd.isna(X.loc[0, "Cholesterol"]))
        self.assertTrue(pd.isna(X.loc[1, "RestingBP"]))

    def test_preprocessor_handles_mixed_features_and_missing_values(self) -> None:
        X, _ = prepare_features_and_target(self.data)
        prepared = build_preprocessor(X).fit_transform(X)

        self.assertEqual(prepared.shape[0], len(X))
        self.assertTrue(np.isfinite(prepared).all())

    def test_unknown_target_name_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No se encontró"):
            prepare_features_and_target(self.data, target_column="target")


if __name__ == "__main__":
    unittest.main()
