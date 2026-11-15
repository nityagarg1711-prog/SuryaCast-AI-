"""Solar geometry and a transparent, simplified rooftop-PV power model."""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEZONE = "Asia/Kolkata"
DEFAULT_LATITUDE = 12.97
DEFAULT_LONGITUDE = 77.59


def solar_position(
    timestamps: pd.DatetimeIndex | list | np.ndarray,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return solar elevation (degrees), azimuth (degrees clockwise from north),
    and cosine of zenith using the NOAA fractional-year approximation.
    """
    if not np.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude must be finite and between -90 and 90 degrees")
    if not np.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be finite and between -180 and 180 degrees")
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if index.hasnans:
        raise ValueError("timestamps must not contain NaT")
    if index.tz is None:
        index = index.tz_localize(TIMEZONE)
    else:
        index = index.tz_convert(TIMEZONE)

    day = index.day_of_year.to_numpy(dtype=float)
    local_hour = (
        index.hour.to_numpy()
        + index.minute.to_numpy() / 60.0
        + index.second.to_numpy() / 3600.0
    )
    fractional_year = 2.0 * np.pi / 365.0 * (
        day - 1.0 + (local_hour - 12.0) / 24.0
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * np.cos(fractional_year)
        - 0.032077 * np.sin(fractional_year)
        - 0.014615 * np.cos(2.0 * fractional_year)
        - 0.040849 * np.sin(2.0 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(fractional_year)
        + 0.070257 * np.sin(fractional_year)
        - 0.006758 * np.cos(2.0 * fractional_year)
        + 0.000907 * np.sin(2.0 * fractional_year)
        - 0.002697 * np.cos(3.0 * fractional_year)
        + 0.00148 * np.sin(3.0 * fractional_year)
    )
    utc_offset_minutes = np.asarray(
        [stamp.utcoffset().total_seconds() / 60.0 for stamp in index], dtype=float
    )
    local_minutes = local_hour * 60.0
    true_solar_minutes = (
        local_minutes + equation_of_time + 4.0 * longitude - utc_offset_minutes
    ) % 1440.0
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude_rad = np.deg2rad(latitude)
    cosine_zenith = np.clip(
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle),
        -1.0,
        1.0,
    )
    elevation = np.rad2deg(np.arcsin(cosine_zenith))
    azimuth = (
        np.rad2deg(
            np.arctan2(
                np.sin(hour_angle),
                np.cos(hour_angle) * np.sin(latitude_rad)
                - np.tan(declination) * np.cos(latitude_rad),
            )
        )
        + 180.0
    ) % 360.0
    return elevation, azimuth, cosine_zenith


def plane_of_array_irradiance(
    timestamps: pd.DatetimeIndex | list | np.ndarray,
    ghi_w_m2,
    dni_w_m2,
    dhi_w_m2,
    panel_tilt_deg: float,
    panel_azimuth_deg: float,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    ground_albedo: float = 0.2,
) -> np.ndarray:
    """Estimate plane-of-array irradiance with beam, isotropic-sky and ground terms.

    GHI is global horizontal irradiance, DNI is direct normal irradiance, and
    DHI is diffuse horizontal irradiance, all in W/m². Panel azimuth is degrees
    clockwise from true north (180° is south). This isotropic diffuse model is
    intentionally simple; it is not a site-specific radiative-transfer model.
    """
    if not 0.0 <= panel_tilt_deg <= 90.0:
        raise ValueError("panel_tilt_deg must be between 0 and 90 degrees")
    if not 0.0 <= panel_azimuth_deg <= 360.0:
        raise ValueError("panel_azimuth_deg must be between 0 and 360 degrees")
    if not 0.0 <= ground_albedo <= 1.0:
        raise ValueError("ground_albedo must be between 0 and 1")

    elevation, sun_azimuth, cosine_zenith = solar_position(
        timestamps, latitude=latitude, longitude=longitude
    )
    ghi, dni, dhi, cosine_zenith = np.broadcast_arrays(
        np.asarray(ghi_w_m2, dtype=float),
        np.asarray(dni_w_m2, dtype=float),
        np.asarray(dhi_w_m2, dtype=float),
        cosine_zenith,
    )
    if not all(np.isfinite(values).all() for values in (ghi, dni, dhi)):
        raise ValueError("GHI, DNI and DHI inputs must be finite")
    elevation_rad = np.deg2rad(elevation)
    tilt_rad = np.deg2rad(panel_tilt_deg)
    incidence_cosine = (
        cosine_zenith * np.cos(tilt_rad)
        + np.cos(elevation_rad)
        * np.sin(tilt_rad)
        * np.cos(np.deg2rad(sun_azimuth - panel_azimuth_deg))
    )
    beam = np.maximum(dni, 0.0) * np.maximum(incidence_cosine, 0.0)
    sky_diffuse = np.maximum(dhi, 0.0) * (1.0 + np.cos(tilt_rad)) / 2.0
    ground_reflected = (
        np.maximum(ghi, 0.0) * ground_albedo * (1.0 - np.cos(tilt_rad)) / 2.0
    )
    poa = np.maximum(beam + sky_diffuse + ground_reflected, 0.0)
    return np.where(elevation > 0.0, poa, 0.0)


def scale_components_to_ghi(
    timestamps: pd.DatetimeIndex | list | np.ndarray,
    ghi_w_m2,
    dni_w_m2,
    dhi_w_m2,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale forecast DNI/DHI so their horizontal sum matches GHI.

    The forecast's direct/diffuse split is preserved when it remains within a
    broad 1600 W/m² DNI guardrail. If the split is absent or would imply
    excessive DNI, the remaining horizontal irradiance is represented as
    diffuse; direct irradiance is reduced if it alone exceeds the requested GHI.
    """
    elevation, _, cosine_zenith = solar_position(
        timestamps, latitude=latitude, longitude=longitude
    )
    ghi, dni, dhi, cosine_zenith = np.broadcast_arrays(
        np.asarray(ghi_w_m2, dtype=float),
        np.asarray(dni_w_m2, dtype=float),
        np.asarray(dhi_w_m2, dtype=float),
        cosine_zenith,
    )
    if not all(np.isfinite(values).all() for values in (ghi, dni, dhi)):
        raise ValueError("GHI, DNI and DHI inputs must be finite")
    ghi = np.maximum(ghi, 0.0)
    dni = np.maximum(dni, 0.0)
    dhi = np.maximum(dhi, 0.0)
    direct_horizontal = dni * np.maximum(cosine_zenith, 0.0)
    component_ghi = direct_horizontal + dhi
    scale = np.divide(
        ghi,
        component_ghi,
        out=np.zeros_like(ghi, dtype=float),
        where=component_ghi > 1e-6,
    )
    scaled_dni = np.minimum(dni * scale, 1600.0)
    direct_horizontal = scaled_dni * np.maximum(cosine_zenith, 0.0)
    direct_exceeds_target = direct_horizontal > ghi
    scaled_dni = np.where(
        direct_exceeds_target,
        np.divide(
            ghi,
            np.maximum(cosine_zenith, 1e-6),
            out=np.zeros_like(ghi, dtype=float),
            where=cosine_zenith > 0.0,
        ),
        scaled_dni,
    )
    scaled_dhi = np.maximum(
        ghi - scaled_dni * np.maximum(cosine_zenith, 0.0), 0.0
    )
    night = elevation <= 0.0
    return np.where(night, 0.0, scaled_dni), np.where(night, 0.0, scaled_dhi)


def estimate_pv_generation(
    timestamps: pd.DatetimeIndex | list | np.ndarray,
    ghi_w_m2,
    dni_w_m2,
    dhi_w_m2,
    air_temperature_c,
    capacity_kwp: float = 5.0,
    panel_tilt_deg: float = 15.0,
    panel_azimuth_deg: float = 180.0,
    system_losses_pct: float = 14.0,
    temperature_coefficient_pct_per_c: float = -0.4,
    interval_hours: float = 1.0,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    noct_c: float = 45.0,
    inverter_efficiency: float = 0.96,
) -> pd.DataFrame:
    """Estimate hourly PV output; outputs are not measured electricity.

    A PVWatts-style linear DC relation is used:
    Pdc = kWp × POA/1000 × [1 + gamma × (Tcell - 25°C)].
    Cell temperature follows the NOCT approximation
    Tcell = Tambient + (NOCT - 20)/800 × POA. Aggregate system losses and a
    fixed inverter efficiency are then applied. The model omits inverter
    clipping, shading, soiling dynamics, wiring topology, and module-specific
    behavior. With hourly mean irradiance, kW × interval-hours gives kWh.
    """
    if capacity_kwp <= 0:
        raise ValueError("capacity_kwp must be positive")
    if not np.isfinite(capacity_kwp):
        raise ValueError("capacity_kwp must be finite")
    if not 0.0 <= system_losses_pct < 100.0:
        raise ValueError("system_losses_pct must be in [0, 100)")
    if not -1.0 <= temperature_coefficient_pct_per_c <= 0.0:
        raise ValueError("temperature coefficient must be between -1 and 0 %/°C")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    if not np.isfinite(interval_hours):
        raise ValueError("interval_hours must be finite")
    if not 0.0 < inverter_efficiency <= 1.0:
        raise ValueError("inverter_efficiency must be in (0, 1]")
    if not np.isfinite(system_losses_pct):
        raise ValueError("system_losses_pct must be finite")
    if not np.isfinite(temperature_coefficient_pct_per_c):
        raise ValueError("temperature coefficient must be finite")
    if not np.isfinite(noct_c) or not np.isfinite(inverter_efficiency):
        raise ValueError("NOCT and inverter efficiency must be finite")

    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    poa = plane_of_array_irradiance(
        index,
        ghi_w_m2,
        dni_w_m2,
        dhi_w_m2,
        panel_tilt_deg=panel_tilt_deg,
        panel_azimuth_deg=panel_azimuth_deg,
        latitude=latitude,
        longitude=longitude,
    )
    air_temperature = np.broadcast_to(
        np.asarray(air_temperature_c, dtype=float), poa.shape
    )
    if not np.isfinite(air_temperature).all():
        raise ValueError("air_temperature_c inputs must be finite")
    cell_temperature = air_temperature + (noct_c - 20.0) / 800.0 * poa
    gamma = temperature_coefficient_pct_per_c / 100.0
    temperature_factor = np.maximum(0.0, 1.0 + gamma * (cell_temperature - 25.0))
    dc_power_kw = capacity_kwp * poa / 1000.0 * temperature_factor
    ac_power_kw = (
        dc_power_kw * (1.0 - system_losses_pct / 100.0) * inverter_efficiency
    )
    energy_kwh = ac_power_kw * interval_hours
    elevation, _, _ = solar_position(index, latitude, longitude)
    night = elevation <= 0.0
    poa = np.where(night, 0.0, poa)
    cell_temperature = np.where(night, air_temperature, cell_temperature)
    ac_power_kw = np.where(night, 0.0, ac_power_kw)
    energy_kwh = np.where(night, 0.0, energy_kwh)
    return pd.DataFrame(
        {
            "poa_irradiance_w_m2": poa,
            "panel_temperature_c": cell_temperature,
            "estimated_power_kw": ac_power_kw,
            "estimated_energy_kwh": energy_kwh,
        },
        index=index,
    )

