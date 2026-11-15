"""SuryaCast AI — next-day solar intelligence for Bengaluru."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_pipeline import (
    DataSourceError,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    fetch_live_forecast,
    fetch_training_data,
)
from forecasting import build_next_24_hour_forecast
from model import ModelAudit, fit_and_evaluate
from solar_physics import estimate_pv_generation, scale_components_to_ghi

st.set_page_config(
    page_title="SuryaCast AI | Solar Intelligence",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #071321; color: #e7f2fb; }
    [data-testid="stSidebar"] { background: #0b1c2b; border-right: 1px solid #17384a; }
    [data-testid="stMetric"] {
      background: linear-gradient(145deg, #10293a, #0c1d2c);
      border: 1px solid #1b4658; padding: 14px 18px; border-radius: 12px;
    }
    [data-testid="stMetricLabel"] { color: #9ab6c6; }
    .hero {
      background: radial-gradient(circle at 85% 15%, #0c5961 0, #103244 32%, #0b1c2b 75%);
      border: 1px solid #1d5b68; border-radius: 18px; padding: 25px 30px;
      margin-bottom: 20px;
    }
    .hero p { color: #b3cbd6; max-width: 860px; margin-bottom: 0; }
    .eyebrow { color: #48e0cc; text-transform: uppercase; letter-spacing: .16em;
      font-size: .73rem; font-weight: 700; }
    h1, h2, h3 { color: #edf8ff; letter-spacing: -.025em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def load_training_history() -> pd.DataFrame:
    return fetch_training_data(
        start_date=date(2023, 1, 1),
        end_date=date.today() - timedelta(days=6),
    )


@st.cache_resource(show_spinner=False, ttl=24 * 60 * 60)
def load_model_audit() -> ModelAudit:
    return fit_and_evaluate(load_training_history())


@st.cache_data(show_spinner=False, ttl=30 * 60)
def load_live_weather() -> pd.DataFrame:
    return fetch_live_forecast()


def _dark_figure(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#0b1c2b",
        plot_bgcolor="#0b1c2b",
        font={"color": "#d9e8f0"},
        height=height,
        margin={"l": 22, "r": 18, "t": 55, "b": 22},
        legend={"orientation": "h", "y": -0.22},
        hoverlabel={"bgcolor": "#10293a", "font_color": "#e7f2fb"},
    )
    fig.update_xaxes(gridcolor="#1a3445", zerolinecolor="#315164")
    fig.update_yaxes(gridcolor="#1a3445", zerolinecolor="#315164")
    return fig


def _generation(
    forecast: pd.DataFrame,
    ghi: np.ndarray,
    settings: dict[str, float],
) -> pd.DataFrame:
    timestamps = forecast.index
    dni, dhi = scale_components_to_ghi(
        timestamps,
        ghi,
        forecast["nwp_dni"].to_numpy(dtype=float),
        forecast["nwp_dhi"].to_numpy(dtype=float),
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )
    return estimate_pv_generation(
        timestamps=timestamps,
        ghi_w_m2=ghi,
        dni_w_m2=dni,
        dhi_w_m2=dhi,
        air_temperature_c=forecast["temperature_2m"].to_numpy(dtype=float),
        capacity_kwp=settings["capacity_kwp"],
        panel_tilt_deg=settings["panel_tilt_deg"],
        panel_azimuth_deg=settings["panel_azimuth_deg"],
        system_losses_pct=settings["system_losses_pct"],
        temperature_coefficient_pct_per_c=settings["temperature_coefficient_pct_per_c"],
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )


def _metric_rows(audit: ModelAudit) -> pd.DataFrame:
    rows = []
    for name, scores in audit.test_metrics.items():
        validation = audit.validation_metrics[name]
        rows.append(
            {
                "Forecast": name,
                "Validation daylight MAE (W/m²)": validation["daylight"]["mae"],
                "Test daylight MAE (W/m²)": scores["daylight"]["mae"],
                "Test daylight RMSE (W/m²)": scores["daylight"]["rmse"],
                "Test daylight R²": scores["daylight"]["r2"],
                "Test all-hour MAE (W/m²)": scores["all_hours"]["mae"],
                "Test all-hour R²": scores["all_hours"]["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values("Test daylight MAE (W/m²)")


def _model_performance_chart(audit: ModelAudit) -> go.Figure:
    history = pd.DataFrame(
        {
            "ERA5 reference": audit.actual_test,
            "Direct NWP": audit.predictions_test["Direct NWP radiation"],
            "Selected source": audit.predictions_test[audit.selected_source],
        }
    ).tail(24 * 7)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["ERA5 reference"],
            name="ERA5 reanalysis reference",
            mode="lines",
            line={"color": "#e5f4ff", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Direct NWP"],
            name="Direct NWP radiation",
            mode="lines",
            line={"color": "#f0ad5a", "width": 1.4, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["Selected source"],
            name=f"Selected: {audit.selected_source}",
            mode="lines",
            line={"color": "#40dec7", "width": 1.8},
        )
    )
    fig.update_layout(title="Chronological holdout · final 7 days")
    fig.update_yaxes(title="Hourly GHI (W/m²)", rangemode="tozero")
    fig.update_xaxes(title="Local time · Asia/Kolkata")
    return _dark_figure(fig, 390)


def _forecast_irradiance_chart(forecast: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if forecast["ghi_lower_w_m2"].notna().all():
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["ghi_upper_w_m2"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["ghi_lower_w_m2"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(64,222,199,0.16)",
                name="Empirical 80% interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["ghi_prediction_w_m2"],
            mode="lines+markers",
            name=f"Selected · {forecast['forecast_source'].iloc[0]}",
            line={"color": "#40dec7", "width": 3},
            marker={"size": 6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["nwp_ghi"],
            mode="lines",
            name="Direct Open-Meteo NWP",
            line={"color": "#f0ad5a", "width": 1.5, "dash": "dash"},
        )
    )
    fig.update_layout(title="Next 24 complete hours · irradiance")
    fig.update_yaxes(title="GHI (W/m²)", rangemode="tozero")
    fig.update_xaxes(title="Local time · Asia/Kolkata", tickformat="%a %H:%M")
    return _dark_figure(fig, 390)


def _energy_chart(
    forecast: pd.DataFrame,
    generation: pd.DataFrame,
    low_generation: pd.DataFrame | None,
    high_generation: pd.DataFrame | None,
) -> go.Figure:
    daylight = forecast["solar_elevation_deg"].to_numpy() > 0
    colors = np.where(daylight, "#22c8b3", "#21384a")
    fig = go.Figure()
    if low_generation is not None and high_generation is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=high_generation["estimated_energy_kwh"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=low_generation["estimated_energy_kwh"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(64,222,199,0.14)",
                name="Empirical range",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Bar(
            x=forecast.index,
            y=generation["estimated_energy_kwh"],
            name="Estimated energy",
            marker_color=colors,
            hovertemplate="%{x|%a %H:%M}<br>%{y:.2f} kWh estimated<extra></extra>",
        )
    )
    fig.update_layout(title="Estimated hourly PV energy")
    fig.update_yaxes(title="Estimated energy (kWh)", rangemode="tozero")
    fig.update_xaxes(title="Local time · Asia/Kolkata", tickformat="%a %H:%M")
    return _dark_figure(fig, 380)


def _sidebar_settings() -> dict[str, float]:
    st.sidebar.markdown('<div class="eyebrow">System design</div>', unsafe_allow_html=True)
    st.sidebar.title("Rooftop configuration")
    capacity = st.sidebar.slider("Installed capacity (kWp DC)", 1.0, 30.0, 5.0, 0.5)
    tilt = st.sidebar.slider("Panel tilt (degrees)", 0, 60, 15, 1)
    azimuth = st.sidebar.slider(
        "Panel azimuth · 0° north, 180° south", 0, 360, 180, 5
    )
    losses = st.sidebar.slider("Aggregate system losses (%)", 0.0, 35.0, 14.0, 1.0)
    temperature_coefficient = st.sidebar.slider(
        "Module temperature coefficient (%/°C)",
        -0.6,
        -0.1,
        -0.4,
        0.05,
        help="Negative power change per degree above 25°C; typical crystalline-silicon modules vary by model.",
    )
    st.sidebar.caption(
        "Generation is a modelled estimate, not meter data. The PV model assumes "
        "NOCT 45°C and 96% inverter efficiency."
    )
    return {
        "capacity_kwp": capacity,
        "panel_tilt_deg": float(tilt),
        "panel_azimuth_deg": float(azimuth),
        "system_losses_pct": losses,
        "temperature_coefficient_pct_per_c": temperature_coefficient,
    }


def main() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Veritas AI · Climate intelligence</div>
          <h1>SuryaCast AI</h1>
          <p>Next-day solar resource forecasting for Bengaluru. A machine-learning
          post-processor is tested against persistence and direct numerical weather
          forecasts before it is trusted to drive a rooftop energy estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    settings = _sidebar_settings()

    try:
        live_weather = load_live_weather()
    except DataSourceError as error:
        st.error(f"Live data could not be loaded: {error}")
        st.info("Check connectivity and try again. No synthetic measurements are substituted.")
        st.stop()

    try:
        with st.spinner("Training and validating chronological solar models…"):
            audit = load_model_audit()
    except DataSourceError as error:
        st.error(f"Historical training data could not be loaded: {error}")
        st.info("No sample or fabricated measurements are substituted.")
        st.stop()
    try:
        forecast = build_next_24_hour_forecast(live_weather, audit)
    except (DataSourceError, ValueError) as error:
        st.error(f"The 24-hour forecast is unavailable: {error}")
        st.stop()

    ghi = forecast["ghi_prediction_w_m2"].to_numpy(dtype=float)
    generation = _generation(forecast, ghi, settings)
    low_generation = None
    high_generation = None
    if forecast["ghi_lower_w_m2"].notna().all():
        low_generation = _generation(
            forecast, forecast["ghi_lower_w_m2"].to_numpy(dtype=float), settings
        )
        high_generation = _generation(
            forecast, forecast["ghi_upper_w_m2"].to_numpy(dtype=float), settings
        )

    daily_energy = float(generation["estimated_energy_kwh"].sum())
    peak_row = generation["estimated_power_kw"].idxmax()
    peak_power = float(generation.loc[peak_row, "estimated_power_kw"])
    useful_sun_hours = int((ghi >= 200.0).sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Estimated next-24h energy", f"{daily_energy:.2f} kWh")
    metric_columns[1].metric(
        "Peak estimated output",
        f"{peak_power:.2f} kW",
        peak_row.strftime("%a %H:%M"),
    )
    metric_columns[2].metric("Configured rooftop", f"{settings['capacity_kwp']:.1f} kWp")
    metric_columns[3].metric("Useful solar hours", f"{useful_sun_hours} h", "GHI ≥ 200 W/m²")
    if low_generation is not None and high_generation is not None:
        st.caption(
            "Empirical 80% interval (test-coverage checked): "
            f"{low_generation['estimated_energy_kwh'].sum():.2f}–"
            f"{high_generation['estimated_energy_kwh'].sum():.2f} kWh. "
            f"Independent daylight holdout coverage: "
            f"{audit.interval_test_coverage:.0%}."
        )
    else:
        st.caption(
            "No generation interval is shown: the independent holdout did not "
            "validate an adequately calibrated 80% daylight interval."
        )

    st.markdown("### 24-hour Bengaluru forecast")
    irradiance_col, energy_col = st.columns(2, gap="medium")
    with irradiance_col:
        st.plotly_chart(
            _forecast_irradiance_chart(forecast),
            width="stretch",
            config={"displaylogo": False},
        )
    with energy_col:
        st.plotly_chart(
            _energy_chart(forecast, generation, low_generation, high_generation),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Irradiance and weather lines are forecast-model outputs. PV power and "
        "hourly energy are physics-based estimates; none are meter measurements."
    )

    st.markdown("### Forecast conditions and energy planning")
    conditions = forecast[
        [
            "nwp_ghi",
            "cloud_cover",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "precipitation",
            "ghi_prediction_w_m2",
        ]
    ].rename(
        columns={
            "nwp_ghi": "Direct NWP GHI (W/m²)",
            "cloud_cover": "Cloud cover (%)",
            "temperature_2m": "Air temperature (°C)",
            "relative_humidity_2m": "Relative humidity (%)",
            "wind_speed_10m": "Wind speed (km/h)",
            "precipitation": "Precipitation (mm)",
            "ghi_prediction_w_m2": "Selected GHI (W/m²)",
        }
    )
    st.dataframe(
        conditions.style.format(precision=1, na_rep="—"),
        width="stretch",
    )
    planning_hours = generation["estimated_energy_kwh"].sort_values(ascending=False).head(4)
    st.markdown("**Highest modelled energy hours**")
    st.write(
        " · ".join(
            f"{hour:%a %H:%M} ({energy:.2f} kWh estimated)"
            for hour, energy in planning_hours.items()
            if energy > 0
        )
        or "No positive generation hours in this forecast window."
    )

    st.markdown("### Model audit · chronological holdout")
    st.write(
        f"Forecast source used: **{audit.selected_source}**. Selection was made "
        "on the validation segment by daylight MAE; the final test period was not "
        "used for model selection."
    )
    metric_table = _metric_rows(audit)
    st.dataframe(
        metric_table.style.format(
            {
                "Validation daylight MAE (W/m²)": "{:.1f}",
                "Test daylight MAE (W/m²)": "{:.1f}",
                "Test daylight RMSE (W/m²)": "{:.1f}",
                "Test daylight R²": "{:.3f}",
                "Test all-hour MAE (W/m²)": "{:.1f}",
                "Test all-hour R²": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Training / calibration / validation / final test timestamps: "
        + " · ".join(f"{key}: {value}" for key, value in audit.split_dates.items())
        + ". Test data is ERA5 reanalysis, a gridded reference—not a local pyranometer."
    )
    st.plotly_chart(
        _model_performance_chart(audit),
        width="stretch",
        config={"displaylogo": False},
    )
    best_ml = min(
        ("Random Forest", "HistGradientBoosting"),
        key=lambda name: audit.test_metrics[name]["daylight"]["mae"],
    )
    nwp_mae = audit.test_metrics["Direct NWP radiation"]["daylight"]["mae"]
    best_ml_mae = audit.test_metrics[best_ml]["daylight"]["mae"]
    if best_ml_mae < nwp_mae:
        st.success(
            f"On the untouched final test period, {best_ml} had lower daylight MAE "
            f"than direct NWP ({best_ml_mae:.1f} vs {nwp_mae:.1f} W/m²). "
            "This is evidence for this test period, not a guarantee of future superiority."
        )
    else:
        st.warning(
            f"No ML candidate beat direct NWP on final-test daylight MAE "
            f"({best_ml_mae:.1f} vs {nwp_mae:.1f} W/m²). The product therefore "
            "uses the validation-selected source and does not claim an AI advantage."
        )

    st.markdown("### Explainability")
    importance_col, association_col = st.columns(2, gap="medium")
    with importance_col:
        importance = audit.feature_importance.head(10).sort_values()
        importance_fig = go.Figure(
            go.Bar(
                x=importance.to_numpy(),
                y=importance.index,
                orientation="h",
                marker_color="#40dec7",
                hovertemplate="%{y}<br>MAE increase: %{x:.2f} W/m²<extra></extra>",
            )
        )
        importance_fig.update_layout(
            title=f"Permutation importance · {audit.explained_model}",
            xaxis_title="Increase in calibration MAE when shuffled (W/m²)",
            yaxis_title="",
        )
        st.plotly_chart(
            _dark_figure(importance_fig, 410),
            width="stretch",
            config={"displaylogo": False},
        )
    with association_col:
        test_features = audit.test_features
        association = pd.DataFrame(
            {
                "Cloud cover (%)": test_features["cloud_cover"],
                "Air temperature (°C)": test_features["temperature_2m"],
                "ERA5 GHI reference (W/m²)": audit.actual_test,
            }
        ).dropna()
        association_fig = go.Figure()
        association_fig.add_trace(
            go.Scatter(
                x=association["Cloud cover (%)"],
                y=association["ERA5 GHI reference (W/m²)"],
                mode="markers",
                marker={"color": "#53a8ff", "size": 5, "opacity": 0.35},
                name="Hourly holdout",
                hovertemplate="Cloud %{x:.0f}%<br>GHI %{y:.0f} W/m²<extra></extra>",
            )
        )
        association_fig.update_layout(
            title="Cloud cover and reference irradiance · descriptive association",
            xaxis_title="Forecast cloud cover (%)",
            yaxis_title="ERA5 reference GHI (W/m²)",
        )
        st.plotly_chart(
            _dark_figure(association_fig, 410),
            width="stretch",
            config={"displaylogo": False},
        )
    st.caption(
        "Permutation importance measures predictive sensitivity on a held-out "
        "calibration segment; weather associations are descriptive, not causal."
    )

    with st.expander("Data provenance, assumptions, and limitations"):
        st.markdown(
            f"""
            - **Location:** Bengaluru, {LATITUDE:.2f}° N, {LONGITUDE:.2f}° E;
              timestamps use `{TIMEZONE}`.
            - **Reference labels:** Open-Meteo Historical Weather API,
              `models=era5`, hourly `shortwave_radiation` (W/m²). These are
              gridded reanalysis estimates, not measured rooftop irradiance.
            - **Training predictors / direct NWP baseline:** Open-Meteo Historical
              Forecast API, including its archived shortwave radiation and weather
              forecasts. The live chart uses Open-Meteo Forecast API data.
            - **PV estimate:** hourly DNI/DHI/GHI plane-of-array approximation,
              NOCT cell-temperature model, user-configurable system losses, and
              assumed 96% inverter efficiency. No clipping, shading, soiling,
              or roof obstructions are modelled.
            - **Uncertainty:** shown only when the separate calibration and final
              chronological test segments support an empirical 80% interval.
              It is not a probabilistic guarantee.
            - **Limits:** reanalysis is not a local instrument; forecast grids
              cannot resolve rooftop microclimate; model skill can vary by season,
              weather regime, and provider model update. The result is not a
              utility yield or savings guarantee.
            """
        )
        st.markdown(
            "[Open-Meteo Forecast API](https://open-meteo.com/en/docs) · "
            "[Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · "
            "[Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)"
        )


if __name__ == "__main__":
    main()

