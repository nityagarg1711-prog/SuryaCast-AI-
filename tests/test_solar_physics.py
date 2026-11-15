import unittest

import numpy as np
import pandas as pd

from solar_physics import (
    estimate_pv_generation,
    plane_of_array_irradiance,
    scale_components_to_ghi,
    solar_position,
)


class SolarPhysicsTests(unittest.TestCase):
    def test_solar_noon_has_high_elevation_near_equator(self):
        timestamp = pd.DatetimeIndex(["2024-03-20 12:00"], tz="UTC")
        elevation, _, cosine_zenith = solar_position(
            timestamp, latitude=0.0, longitude=0.0
        )
        self.assertGreater(float(elevation[0]), 85.0)
        self.assertGreater(float(cosine_zenith[0]), 0.99)

    def test_horizontal_plane_matches_consistent_ghi_components(self):
        timestamp = pd.DatetimeIndex(["2024-03-20 12:00"], tz="UTC")
        poa = plane_of_array_irradiance(
            timestamp,
            ghi_w_m2=[600.0],
            dni_w_m2=[500.0],
            dhi_w_m2=[100.0],
            panel_tilt_deg=0.0,
            panel_azimuth_deg=180.0,
            latitude=0.0,
            longitude=0.0,
        )
        self.assertAlmostEqual(float(poa[0]), 600.0, delta=2.0)

    def test_component_scaling_preserves_horizontal_ghi(self):
        timestamp = pd.date_range(
            "2024-03-20 08:00", periods=4, freq="h", tz="Asia/Kolkata"
        )
        target_ghi = np.array([100.0, 300.0, 500.0, 200.0])
        dni, dhi = scale_components_to_ghi(
            timestamp,
            target_ghi,
            dni_w_m2=[200.0, 400.0, 600.0, 300.0],
            dhi_w_m2=[60.0, 100.0, 120.0, 80.0],
        )
        _, _, cosine_zenith = solar_position(timestamp)
        reconstructed = dni * np.maximum(cosine_zenith, 0.0) + dhi
        np.testing.assert_allclose(reconstructed, target_ghi, atol=1e-8)

    def test_component_scaling_bounds_dni_without_losing_target_ghi(self):
        timestamp = pd.DatetimeIndex(["2024-06-21 07:00"], tz="Asia/Kolkata")
        target_ghi = np.array([1200.0])
        dni, dhi = scale_components_to_ghi(
            timestamp,
            target_ghi,
            dni_w_m2=[1000.0],
            dhi_w_m2=[0.0],
        )
        _, _, cosine_zenith = solar_position(timestamp)
        self.assertLessEqual(float(dni[0]), 1600.0)
        np.testing.assert_allclose(
            dni * np.maximum(cosine_zenith, 0.0) + dhi,
            target_ghi,
            atol=1e-8,
        )

    def test_nighttime_generation_is_exactly_zero(self):
        timestamps = pd.date_range(
            "2024-06-21 00:00", periods=3, freq="h", tz="Asia/Kolkata"
        )
        result = estimate_pv_generation(
            timestamps,
            ghi_w_m2=[0.0, 0.0, 0.0],
            dni_w_m2=[0.0, 0.0, 0.0],
            dhi_w_m2=[0.0, 0.0, 0.0],
            air_temperature_c=[23.0, 23.0, 23.0],
            capacity_kwp=5.0,
        )
        np.testing.assert_array_equal(result["estimated_energy_kwh"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["estimated_power_kw"], [0.0, 0.0, 0.0])

    def test_generation_scales_with_capacity_and_sums_hourly_energy(self):
        timestamps = pd.date_range(
            "2024-03-20 08:00", periods=3, freq="h", tz="Asia/Kolkata"
        )
        kwargs = {
            "timestamps": timestamps,
            "ghi_w_m2": [500.0, 700.0, 600.0],
            "dni_w_m2": [450.0, 600.0, 500.0],
            "dhi_w_m2": [100.0, 120.0, 110.0],
            "air_temperature_c": [25.0, 30.0, 32.0],
        }
        five_kw = estimate_pv_generation(**kwargs, capacity_kwp=5.0)
        ten_kw = estimate_pv_generation(**kwargs, capacity_kwp=10.0)
        self.assertGreater(five_kw["estimated_energy_kwh"].sum(), 0.0)
        self.assertAlmostEqual(
            float(ten_kw["estimated_energy_kwh"].sum()),
            2.0 * float(five_kw["estimated_energy_kwh"].sum()),
            places=10,
        )
        self.assertAlmostEqual(
            float(five_kw["estimated_energy_kwh"].sum()),
            float(five_kw["estimated_power_kw"].sum()),
            places=10,
        )

    def test_roof_orientation_losses_and_temperature_coefficient_change_output(self):
        timestamp = pd.DatetimeIndex(["2024-03-20 08:00"], tz="Asia/Kolkata")
        base = {
            "timestamps": timestamp,
            "ghi_w_m2": [600.0],
            "dni_w_m2": [700.0],
            "dhi_w_m2": [50.0],
            "air_temperature_c": [35.0],
            "capacity_kwp": 5.0,
            "panel_tilt_deg": 30.0,
        }
        east = estimate_pv_generation(**base, panel_azimuth_deg=90.0)
        west = estimate_pv_generation(**base, panel_azimuth_deg=270.0)
        self.assertGreater(
            east["estimated_energy_kwh"].iloc[0],
            west["estimated_energy_kwh"].iloc[0],
        )

        no_losses = estimate_pv_generation(
            **base, panel_azimuth_deg=90.0, system_losses_pct=0.0
        )
        twenty_percent_losses = estimate_pv_generation(
            **base, panel_azimuth_deg=90.0, system_losses_pct=20.0
        )
        self.assertAlmostEqual(
            twenty_percent_losses["estimated_energy_kwh"].iloc[0],
            0.8 * no_losses["estimated_energy_kwh"].iloc[0],
            places=10,
        )
        hotter_coefficient = estimate_pv_generation(
            **base,
            panel_azimuth_deg=90.0,
            temperature_coefficient_pct_per_c=-0.5,
        )
        milder_coefficient = estimate_pv_generation(
            **base,
            panel_azimuth_deg=90.0,
            temperature_coefficient_pct_per_c=-0.1,
        )
        self.assertLess(
            hotter_coefficient["estimated_energy_kwh"].iloc[0],
            milder_coefficient["estimated_energy_kwh"].iloc[0],
        )


if __name__ == "__main__":
    unittest.main()
