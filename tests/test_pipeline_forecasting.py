import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from data_pipeline import DataSourceError, _cached_json, _hourly_frame
from forecasting import build_next_24_hour_forecast


class FakeAudit:
    selected_source = "Direct NWP radiation"
    interval_reliable = False
    interval_residuals = None

    def predict(self, features):
        return features["nwp_ghi"].to_numpy(dtype=float)


class PipelineForecastTests(unittest.TestCase):
    def test_api_hourly_payload_is_localized_and_validated(self):
        payload = {
            "hourly": {
                "time": ["2025-01-01T00:00", "2025-01-01T01:00"],
                "shortwave_radiation": [0.0, 12.0],
            }
        }
        result = _hourly_frame(payload, ("shortwave_radiation",))
        self.assertEqual(str(result.index.tz), "Asia/Kolkata")
        self.assertEqual(result["shortwave_radiation"].tolist(), [0.0, 12.0])
        with self.assertRaises(DataSourceError):
            _hourly_frame(payload, ("cloud_cover",))

    def test_json_cache_avoids_repeated_requests(self):
        response = Mock()
        response.json.return_value = {"hourly": {"time": []}}
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"SURYACAST_CACHE_DIR": directory}):
                with patch("data_pipeline.requests.get", return_value=response) as get:
                    first = _cached_json(
                        "https://example.invalid/data",
                        {"latitude": 12.97},
                        ttl_seconds=60,
                    )
                    second = _cached_json(
                        "https://example.invalid/data",
                        {"latitude": 12.97},
                        ttl_seconds=60,
                    )
        self.assertEqual(first, second)
        get.assert_called_once()

    def test_next_day_forecast_has_24_complete_hours_and_zeroes_at_night(self):
        index = pd.date_range(
            "2025-01-01 00:00", periods=120, freq="h", tz="Asia/Kolkata"
        )
        weather = pd.DataFrame(
            {
                "nwp_ghi": 250.0,
                "nwp_dni": 400.0,
                "nwp_dhi": 80.0,
                "cloud_cover": 40.0,
                "temperature_2m": 25.0,
                "relative_humidity_2m": 60.0,
                "wind_speed_10m": 8.0,
                "precipitation": 0.0,
            },
            index=index,
        )
        result = build_next_24_hour_forecast(
            weather,
            FakeAudit(),
            now=pd.Timestamp("2025-01-03 10:30", tz="Asia/Kolkata"),
        )
        self.assertEqual(len(result), 24)
        self.assertEqual(result.index[0], pd.Timestamp("2025-01-03 11:00", tz="Asia/Kolkata"))
        self.assertTrue((result.loc[result["solar_elevation_deg"] <= 0, "ghi_prediction_w_m2"] == 0).all())
        self.assertTrue(np.isfinite(result["ghi_prediction_w_m2"]).all())

    def test_missing_forecast_hours_raise_explicit_error(self):
        index = pd.date_range(
            "2025-01-01 00:00", periods=72, freq="h", tz="Asia/Kolkata"
        ).delete([60])
        weather = pd.DataFrame(
            {
                "nwp_ghi": 250.0,
                "nwp_dni": 400.0,
                "nwp_dhi": 80.0,
                "cloud_cover": 40.0,
                "temperature_2m": 25.0,
                "relative_humidity_2m": 60.0,
                "wind_speed_10m": 8.0,
                "precipitation": 0.0,
            },
            index=index,
        )
        with self.assertRaises(DataSourceError):
            build_next_24_hour_forecast(
                weather,
                FakeAudit(),
                now=pd.Timestamp("2025-01-03 10:30", tz="Asia/Kolkata"),
            )


if __name__ == "__main__":
    unittest.main()
