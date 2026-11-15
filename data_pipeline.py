"""Open-Meteo data access with explicit provenance and a small disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

LATITUDE = 12.97
LONGITUDE = 77.59
TIMEZONE = "Asia/Kolkata"
OBSERVATION_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OBSERVATION_MODEL = "era5"

WEATHER_VARIABLES = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
)

HINDCAST_RENAME = {
    "shortwave_radiation": "nwp_ghi",
    "direct_normal_irradiance": "nwp_dni",
    "diffuse_radiation": "nwp_dhi",
    "cloud_cover": "cloud_cover",
    "temperature_2m": "temperature_2m",
    "relative_humidity_2m": "relative_humidity_2m",
    "wind_speed_10m": "wind_speed_10m",
    "precipitation": "precipitation",
}


class DataSourceError(RuntimeError):
    """Raised when an external data source is unavailable or malformed."""


logger = logging.getLogger(__name__)


def _cache_directory() -> Path:
    directory = Path(
        os.environ.get(
            "SURYACAST_CACHE_DIR",
            Path.home() / ".cache" / "suryacast",
        )
    ).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory



def _cached_json(
    url: str,
    params: dict[str, Any],
    ttl_seconds: int,
    timeout: tuple[float, float] = (10, 60),
) -> dict[str, Any]:
    cache_key = json.dumps([url, params], sort_keys=True, separators=(",", ":"))
    cache_path = _cache_directory() / f"{hashlib.sha256(cache_key.encode()).hexdigest()}.json"

    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < ttl_seconds:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_path.unlink(missing_ok=True)

    payload = None

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException as error:
            if attempt == 2:
                raise DataSourceError(
                    f"Open-Meteo request failed after 3 attempts: {error}"
                ) from error
            time.sleep(2 ** attempt)
        except ValueError as error:
            raise DataSourceError(
                "Open-Meteo returned invalid JSON."
            ) from error

    if not isinstance(payload, dict) or payload.get("error"):
        reason = (
            payload.get("reason", "unexpected response shape")
            if isinstance(payload, dict)
            else "unexpected response shape"
        )
        raise DataSourceError(f"Open-Meteo rejected the request: {reason}")

    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=cache_path.parent, delete=False
        ) as temporary_file:
            json.dump(payload, temporary_file)
            temporary_path = temporary_file.name

        os.replace(temporary_path, cache_path)

    except OSError:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)
        logger.warning(
            "Could not write Open-Meteo response cache at %s", cache_path
        )

    return payload



def _hourly_frame(
    payload: dict[str, Any],
    variables: tuple[str, ...],
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise DataSourceError("Open-Meteo response has no hourly time series.")
    times = hourly["time"]
    values: dict[str, list[Any]] = {}
    for variable in variables:
        column = hourly.get(variable)
        if not isinstance(column, list) or len(column) != len(times):
            raise DataSourceError(
                f"Open-Meteo response is missing hourly variable '{variable}'."
            )
        values[variable] = column
    try:
        index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise"))
    except (TypeError, ValueError) as error:
        raise DataSourceError("Open-Meteo returned invalid hourly timestamps.") from error
    if index.tz is None:
        index = index.tz_localize(TIMEZONE)
    else:
        index = index.tz_convert(TIMEZONE)
    frame = pd.DataFrame(values, index=index)
    frame.index.name = "time"
    return frame


def _date_chunks(start: date, end: date):
    for year in range(start.year, end.year + 1):
        yield max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def fetch_training_data(
    start_date: date = date(2023, 1, 1),
    end_date: date | None = None,
) -> pd.DataFrame:
    """Join ERA5 radiation references to archived NWP forecasts.

    Historical ERA5 values are reanalysis estimates, not ground-station
    measurements. The weather predictors come from Open-Meteo's archived
    forecast endpoint so the training inputs resemble operational forecasts.
    """
    latest_complete_date = date.today() - timedelta(days=6)
    end = min(end_date or latest_complete_date, latest_complete_date)
    if start_date > end:
        raise ValueError("start_date must be on or before the latest complete date")

    joined: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_date, end):
        common = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
        }
        reference_payload = _cached_json(
            OBSERVATION_URL,
            {
                **common,
                "hourly": "shortwave_radiation",
                "models": OBSERVATION_MODEL,
            },
            ttl_seconds=30 * 24 * 60 * 60,
        )
        forecast_payload = _cached_json(
            HISTORICAL_FORECAST_URL,
            {**common, "hourly": ",".join(WEATHER_VARIABLES)},
            ttl_seconds=30 * 24 * 60 * 60,
        )
        reference = _hourly_frame(reference_payload, ("shortwave_radiation",))
        forecast = _hourly_frame(forecast_payload, WEATHER_VARIABLES).rename(
            columns=HINDCAST_RENAME
        )
        joined.append(
            reference.rename(columns={"shortwave_radiation": "observed_ghi"}).join(
                forecast, how="inner"
            )
        )
    if not joined:
        raise DataSourceError("No matching historical solar data was returned.")
    result = pd.concat(joined).sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    if result.empty:
        raise DataSourceError("Historical observations and forecasts did not overlap.")
    return result


def fetch_live_forecast() -> pd.DataFrame:
    """Retrieve a compact live forecast for Bengaluru."""
    payload = _cached_json(
        LIVE_FORECAST_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "timezone": TIMEZONE,
            "hourly": ",".join(WEATHER_VARIABLES),
            "past_days": 3,
            "forecast_days": 2,
        },
        ttl_seconds=30 * 60,
        timeout=(10, 30),
    )

    forecast = _hourly_frame(payload, WEATHER_VARIABLES).rename(
        columns=HINDCAST_RENAME
    )

    if forecast.empty:
        raise DataSourceError("Open-Meteo returned an empty live forecast.")

    return forecast
