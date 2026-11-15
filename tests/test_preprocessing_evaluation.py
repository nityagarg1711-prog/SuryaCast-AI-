import unittest

import numpy as np
import pandas as pd

from evaluation import (
    chronological_partitions,
    empirical_residual_interval,
    interval_coverage,
    regression_metrics,
)
from preprocessing import FEATURE_COLUMNS, TARGET, prepare_supervised_frame


def sample_history(length=120):
    index = pd.date_range(
        "2025-01-01", periods=length, freq="h", tz="Asia/Kolkata"
    )
    ramp = np.arange(length, dtype=float)
    return pd.DataFrame(
        {
            TARGET: 100.0 + ramp % 500,
            "nwp_ghi": 120.0 + ramp % 450,
            "nwp_dni": 150.0 + ramp % 700,
            "nwp_dhi": 30.0 + ramp % 150,
            "cloud_cover": 50.0,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 65.0,
            "wind_speed_10m": 8.0,
            "precipitation": 0.0,
        },
        index=index,
    )


class PreprocessingEvaluationTests(unittest.TestCase):
    def test_features_use_nwp_lags_and_never_target_values(self):
        source = sample_history()
        features, target = prepare_supervised_frame(source)
        first_timestamp = source.index[48]
        self.assertAlmostEqual(
            features.loc[first_timestamp, "nwp_ghi_lag_24"],
            source.loc[source.index[24], "nwp_ghi"],
        )
        self.assertAlmostEqual(
            features.loc[first_timestamp, "nwp_ghi_lag_48"],
            source.loc[source.index[0], "nwp_ghi"],
        )
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertNotIn(TARGET, features.columns)
        changed_target = source.copy()
        changed_target.iloc[-1, changed_target.columns.get_loc(TARGET)] = 1500.0
        changed_features, _ = prepare_supervised_frame(changed_target)
        pd.testing.assert_frame_equal(features, changed_features)
        self.assertEqual(len(features), len(target))

    def test_invalid_target_is_dropped_not_filled(self):
        source = sample_history()
        missing_time = source.index[70]
        outlier_time = source.index[71]
        source.loc[missing_time, TARGET] = np.nan
        source.loc[outlier_time, TARGET] = 1700.0
        _, target = prepare_supervised_frame(source)
        self.assertNotIn(missing_time, target.index)
        self.assertNotIn(outlier_time, target.index)

    def test_chronological_partitions_do_not_overlap(self):
        parts = chronological_partitions(1000)
        self.assertEqual(parts["train"], slice(0, 600))
        self.assertEqual(parts["calibration"], slice(600, 700))
        self.assertEqual(parts["validation"], slice(700, 800))
        self.assertEqual(parts["test"], slice(800, 1000))

    def test_regression_metrics_and_empirical_interval(self):
        actual = np.array([0.0, 1.0, 2.0])
        predicted = np.array([0.0, 2.0, 2.0])
        metrics = regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(metrics["r2"], 0.5)

        calibration = np.arange(200, dtype=float)
        bounds = empirical_residual_interval(
            calibration, calibration - 2.0, minimum_samples=100
        )
        self.assertIsNotNone(bounds)
        self.assertAlmostEqual(bounds[0], 2.0)
        self.assertAlmostEqual(bounds[1], 2.0)
        self.assertEqual(interval_coverage([1.0, 2.0], [1.0, 0.0], [1.0, 3.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
