# SuryaCast AI

**Next-day solar resource forecasting and rooftop PV planning for Bengaluru.**

SuryaCast AI compares two scikit-learn models with persistence and direct
numerical-weather-prediction (NWP) radiation. It chooses the live forecast source
using a chronological validation period, reports a separate final test period,
and converts the selected hourly irradiance forecast into a documented
physics-based electricity estimate.

## Run

Python 3.10+ and an internet connection are required for first-time data
retrieval.

```bash
cd SuryaCastAI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The initial run retrieves and caches hourly history from 2023 onward, then
trains and evaluates the models. Open-Meteo's free endpoints can take time to
download several years of data. The application stores raw API responses under
`~/.cache/suryacast` by default; set `SURYACAST_CACHE_DIR` to change this path.
Historical response caches last 30 days and the live forecast cache lasts
30 minutes. A source failure is shown explicitly; no demo observations are
substituted.

Run the focused automated tests with:

```bash
python -m unittest discover -s tests -v
```

## Data sources and provenance

| Purpose | Endpoint | Variables / interpretation |
|---|---|---|
| Current forecast | [Open-Meteo Forecast API](https://open-meteo.com/en/docs) | Hourly shortwave radiation (GHI proxy), direct normal irradiance, diffuse radiation, cloud cover, temperature, humidity, wind, and precipitation. |
| Historical labels | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | `models=era5`, hourly `shortwave_radiation`. This is a gridded **reanalysis estimate**, not a ground pyranometer measurement. |
| Training NWP predictors | [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | Archived forecast-model radiation and weather covariates, time-aligned with ERA5 labels. |

All queries use approximately 12.97° N, 77.59° E and `Asia/Kolkata`. API
timestamps are localized to India Standard Time; radiation is handled in W/m²,
temperature in °C, cloud cover and humidity in percent, wind speed in km/h,
and precipitation in mm. Open-Meteo's archive and historical-forecast
endpoints were live-request verified for these coordinates and variables on
2026-11-15. Endpoint responses, not hard-coded sample measurements, feed the
application.

The historical forecast archive is a best-match NWP time series. Its model
selection and initialization schedule can differ from a fixed-issue 24-hour
hindcast. This prototype therefore tests post-processing of archived NWP
covariates against ERA5 reanalysis; it does not claim pyranometer-verified forecast
skill or a fully fixed-issue operational backtest. The Historical Forecast API
documents archived forecasts from around 2022 onward. The PVWatts service is
not required or called; its optional reference comparison is deferred.

## Machine-learning design

- **Target:** ERA5 hourly `shortwave_radiation` in W/m².
- **Inputs:** historical forecast GHI, DNI, DHI, cloud cover, temperature,
  relative humidity, wind, precipitation, 24/48-hour NWP-GHI lags, local-time
  and annual cycles, and solar geometry.
- **Models:** `RandomForestRegressor` and
  `HistGradientBoostingRegressor`.
- **Baselines:** direct archived NWP GHI and previous-day same-hour ERA5
  GHI persistence. Persistence is for retrospective comparison only; delayed
  reanalysis is not represented as a live rooftop observation.
- **Splitting:** chronological 60% train / 10% interval calibration / 10%
  validation / final 20% test. Models fit only on the earliest block. Validation
  daylight MAE selects between the two ML candidates and direct NWP; the test
  block does not select the live forecast source.
- **Scoring:** MAE, RMSE and R² on all hours and daylight hours. Daylight results
  are emphasized so the many nighttime zeroes do not conceal daytime error.
- **Missingness / quality:** target gaps are dropped, never interpolated.
  Weather predictors may be forward-filled for at most two hours. Negative
  irradiance and values above a broad 1600 W/m² physical guardrail are excluded,
  not winsorized. Modelled NWP predictors, not future observed targets, supply
  the lag features.
- **Uncertainty:** an empirical 80% residual interval is calibrated on its own
  chronological segment and displayed only if daylight coverage on the final
  test segment is between 65% and 95%. This is a calibration check, not a
  probabilistic guarantee.

If direct NWP has lower validation daylight MAE than both ML candidates, the
dashboard uses direct NWP as the live irradiance source and says so. The ML
candidates and their untouched final-test scores remain visible. A model is not
described as superior merely because it is an ML model.

### Verified evaluation snapshot

The 2026-11-15 run used 33,768 usable hourly records after lag construction:
20,260 train, 3,376 calibration, 3,376 validation, and 6,756 final-test rows.
The untouched test window was 2026-02-01 12:00 through 2026-11-09 23:00
Asia/Kolkata. Daylight-only scores against ERA5 reanalysis were:

| Forecast source | MAE (W/m²) | RMSE (W/m²) | R² |
|---|---:|---:|---:|
| Random Forest | 54.59 | 84.39 | 0.916 |
| HistGradientBoosting | 55.50 | 86.53 | 0.912 |
| Direct NWP radiation | 68.28 | 111.66 | 0.853 |
| Previous-day-hour persistence | 72.26 | 114.69 | 0.845 |

Random Forest also had the lowest validation daylight MAE (54.09 W/m²) and was
selected for the live forecast. The nominal 80% empirical interval had 84.2%
daylight coverage on the final test period, so it passed the app's documented
65–95% reliability gate. These are reproducible metrics for this data snapshot,
not instrument-verified site accuracy or a guarantee of future performance.

## PV estimation

The PV model uses an isotropic-sky plane-of-array approximation:

1. Beam irradiance: DNI × positive cosine of incidence.
2. Sky diffuse: DHI × (1 + cos(tilt)) / 2.
3. Ground-reflected: GHI × 0.2 albedo × (1 − cos(tilt)) / 2.
4. Cell temperature: `Tair + (NOCT − 20) / 800 × POA`, with assumed NOCT 45°C.
5. DC power: `capacity_kWp × POA/1000 × [1 + gamma × (Tcell − 25°C)]`.
6. Apply configured aggregate losses and assumed 96% inverter efficiency.
   Hourly mean power in kW multiplied by the one-hour interval gives kWh.

The default temperature coefficient is −0.4%/°C. Nighttime output is explicitly
zero. The estimate does not include inverter clipping, shading, soiling,
module-specific behavior, roof obstructions, or measured system performance.

## Product and climate relevance

The dashboard offers a 24-hour forecast, irradiance and estimated-energy
charts, a configurable rooftop calculator, model comparisons, error metrics,
permutation importance, weather-feature associations, and high-generation
hours for scheduling. These tools can help households, schools, and community
facilities plan flexible electricity use. No savings, emissions reductions,
or real generation are claimed without site and meter validation.

## Strict audit notes and future work

The strongest technical differentiator is the **honest model-selection gate**:
ML must beat direct NWP on validation before it can become the operational
source, and the final chronological block remains a separate check. Solar
geometry and configurable rooftop assumptions then translate that forecast to
an explicitly estimated PV yield.

Limitations include reanalysis rather than local measured labels, historical
NWP model changes, single-location training, coarse gridded weather, the
simplified PV conversion model, and potential seasonal regime changes. Improve
with local pyranometer and inverter data, fixed-issue hindcasts at a chosen
lead time, season-stratified evaluation, a validated PVWatts/pvlib comparison,
roof shading and clipping inputs, and additional Bengaluru neighborhoods.

## Five-slide presentation outline

1. **Problem:** rooftop solar users need an hourly next-day resource estimate;
   distinguish the need from claims about actual generation.
2. **Solution:** SuryaCast AI forecast, explainable comparison, and configurable
   rooftop estimate; demonstrate one Bengaluru day.
3. **Architecture:** archived/live Open-Meteo NWP, ERA5 reference labels,
   chronological model comparison, selection gate, and PV physics.
4. **Validation:** show the final-test MAE/RMSE/R² table for
   persistence, direct NWP, Random Forest and HistGradientBoosting; state whether
   ML won and emphasize the reanalysis—not instrument—reference. The current
   verified snapshot is listed above.
5. **Impact and next steps:** practical energy-planning use, honest limitations,
   and a plan for pyranometer/inverter validation and fixed-lead backtesting.

