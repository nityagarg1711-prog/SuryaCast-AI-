"""Causal feature engineering and quality checks for hourly solar data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import TIMEZONE
from solar_physics import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, solar_position

TARGET = "observed_ghi"
FEATURE_COLUMNS = (
    "nwp_ghi",
    "nwp_dni",
    "nwp_dhi",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_cos_zenith",
    "nwp_ghi_lag_24",
    "nwp_ghi_lag_48",
)


def normalize_hourly_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort and regularize timestamps without filling the target variable."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("input data must have a DatetimeIndex")
    result = frame.copy().sort_index()
    if result.index.tz is None:
        result.index = result.index.tz_localize(TIMEZONE)
    else:
        result.index = result.index.tz_convert(TIMEZONE)
    result = result.loc[~result.index.duplicated(keep="last")]
    return result.asfreq("h")


def engineer_predictors(
    frame: pd.DataFrame,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """Build only inference-available weather, solar-geometry and NWP lag features."""
    data = normalize_hourly_frame(frame)
    required = (
        "nwp_ghi",
        "nwp_dni",
        "nwp_dhi",
        "cloud_cover",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
    )
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"missing forecast variables: {', '.join(missing_columns)}")

    weather = data.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    weather["nwp_ghi"] = weather["nwp_ghi"].where(
        weather["nwp_ghi"].between(0.0, 1600.0)
    )
    weather["nwp_dni"] = weather["nwp_dni"].where(
        weather["nwp_dni"].between(0.0, 1800.0)
    )
    weather["nwp_dhi"] = weather["nwp_dhi"].where(
        weather["nwp_dhi"].between(0.0, 1600.0)
    )
    weather["cloud_cover"] = weather["cloud_cover"].where(
        weather["cloud_cover"].between(0.0, 100.0)
    )
    weather["temperature_2m"] = weather["temperature_2m"].where(
        weather["temperature_2m"].between(-60.0, 70.0)
    )
    weather["relative_humidity_2m"] = weather["relative_humidity_2m"].where(
        weather["relative_humidity_2m"].between(0.0, 100.0)
    )
    weather["wind_speed_10m"] = weather["wind_speed_10m"].where(
        weather["wind_speed_10m"].between(0.0, 200.0)
    )
    weather["precipitation"] = weather["precipitation"].where(
        weather["precipitation"].between(0.0, 500.0)
    )

    # Forward-fill only brief predictor gaps; target values are never imputed.
    weather = weather.ffill(limit=2)
    hour = data.index.hour.to_numpy() + data.index.minute.to_numpy() / 60.0
    day_of_year = data.index.day_of_year.to_numpy(dtype=float)
    elevation, _, cosine_zenith = solar_position(
        data.index, latitude=latitude, longitude=longitude
    )
    predictors = weather.copy()
    predictors["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    predictors["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    predictors["day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    predictors["day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    predictors["solar_cos_zenith"] = np.maximum(cosine_zenith, 0.0)
    predictors["nwp_ghi_lag_24"] = predictors["nwp_ghi"].shift(24)
    predictors["nwp_ghi_lag_48"] = predictors["nwp_ghi"].shift(48)
    predictors = predictors.replace([np.inf, -np.inf], np.nan)
    predictors.attrs["solar_elevation_deg"] = elevation
    return predictors.loc[:, list(FEATURE_COLUMNS)]


def prepare_supervised_frame(
    frame: pd.DataFrame,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return finite model features and ERA5 reanalysis reference labels.

    Negative reference irradiance and values above a deliberately broad
    1600 W/m² physical guardrail are excluded, not clipped or imputed. Weather
    predictor gaps may only be forward-filled for up to two hours.
    """
    data = normalize_hourly_frame(frame)
    if TARGET not in data.columns:
        raise ValueError(f"input data must include '{TARGET}'")
    target = pd.to_numeric(data[TARGET], errors="coerce")
    target = target.where(target.between(0.0, 1600.0))
    predictors = engineer_predictors(
        data, latitude=latitude, longitude=longitude
    )
    valid = target.notna() & predictors.notna().all(axis=1)
    return predictors.loc[valid].copy(), target.loc[valid].astype(float)

