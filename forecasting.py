"""Operational next-24-hour forecasts using live NWP covariates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline import DataSourceError, TIMEZONE
from model import ModelAudit
from preprocessing import engineer_predictors
from solar_physics import DEFAULT_LATITUDE, DEFAULT_LONGITUDE, solar_position


def build_next_24_hour_forecast(
    weather_data: pd.DataFrame,
    model_audit: ModelAudit,
    now: pd.Timestamp | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> pd.DataFrame:
    """Return the next 24 complete local hourly intervals.

    The model uses forecast-issued weather variables and recent NWP context,
    never future observed radiation. At night the physical GHI forecast is
    constrained to zero.
    """
    if not isinstance(weather_data.index, pd.DatetimeIndex):
        raise ValueError("weather_data must have a DatetimeIndex")
    weather = weather_data.sort_index().copy()
    if weather.index.tz is None:
        weather.index = weather.index.tz_localize(TIMEZONE)
    else:
        weather.index = weather.index.tz_convert(TIMEZONE)
    current_time = pd.Timestamp.now(tz=TIMEZONE) if now is None else pd.Timestamp(now)
    if current_time.tz is None:
        current_time = current_time.tz_localize(TIMEZONE)
    else:
        current_time = current_time.tz_convert(TIMEZONE)
    first_hour = current_time.floor("h") + pd.Timedelta(hours=1)
    future_index = weather.index[weather.index >= first_hour][:24]
    if len(future_index) != 24:
        raise DataSourceError(
            f"Only {len(future_index)} future hourly forecast rows are available; 24 are required."
        )
    hourly_intervals = future_index[1:] - future_index[:-1]
    if not (hourly_intervals == pd.Timedelta(hours=1)).all():
        raise DataSourceError("The live forecast has a gap in its next-24-hour timestamps.")

    features = engineer_predictors(weather, latitude=latitude, longitude=longitude)
    future_features = features.reindex(future_index)
    if future_features.isna().any().any():
        missing_count = int(future_features.isna().any(axis=1).sum())
        raise DataSourceError(
            f"Forecast predictors are incomplete for {missing_count} of the next 24 hours."
        )
    result = weather.reindex(future_index).copy()
    prediction = model_audit.predict(future_features)
    elevation, _, _ = solar_position(future_index, latitude, longitude)
    daylight = elevation > 0.0
    prediction = np.where(daylight, np.maximum(prediction, 0.0), 0.0)
    result["solar_elevation_deg"] = elevation
    result["ghi_prediction_w_m2"] = prediction
    result["forecast_source"] = model_audit.selected_source

    result["ghi_lower_w_m2"] = np.nan
    result["ghi_upper_w_m2"] = np.nan
    if model_audit.interval_reliable and model_audit.interval_residuals is not None:
        residual_low, residual_high = model_audit.interval_residuals
        lower = np.maximum(0.0, prediction + residual_low)
        upper = np.maximum(lower, prediction + residual_high)
        result.loc[daylight, "ghi_lower_w_m2"] = lower[daylight]
        result.loc[daylight, "ghi_upper_w_m2"] = upper[daylight]
        result.loc[~daylight, ["ghi_lower_w_m2", "ghi_upper_w_m2"]] = 0.0
    return result

