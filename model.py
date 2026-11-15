"""Chronological model comparison against persistence and direct NWP radiation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from evaluation import (
    chronological_partitions,
    empirical_residual_interval,
    interval_coverage,
    score_overall_and_daylight,
)
from preprocessing import FEATURE_COLUMNS, TARGET, normalize_hourly_frame, prepare_supervised_frame

MODEL_NAMES = ("Random Forest", "HistGradientBoosting")
PERSISTENCE_NAME = "Persistence (previous-day hour)"
NWP_NAME = "Direct NWP radiation"


@dataclass
class ModelAudit:
    models: dict[str, object]
    feature_names: tuple[str, ...]
    selected_source: str
    explained_model: str
    validation_metrics: dict[str, dict[str, dict[str, float]]]
    test_metrics: dict[str, dict[str, dict[str, float]]]
    actual_test: pd.Series
    test_features: pd.DataFrame
    predictions_test: pd.DataFrame
    interval_residuals: tuple[float, float] | None
    interval_test_coverage: float | None
    interval_reliable: bool
    feature_importance: pd.Series
    split_dates: dict[str, str]
    sample_counts: dict[str, int]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        daylight = _daylight_mask(features)
        if self.selected_source == NWP_NAME:
            prediction = np.maximum(features["nwp_ghi"].to_numpy(dtype=float), 0.0)
        else:
            if self.selected_source not in self.models:
                raise ValueError(f"unknown selected forecast source: {self.selected_source}")
            prediction = np.maximum(
                self.models[self.selected_source].predict(
                    features.loc[:, list(self.feature_names)]
                ),
                0.0,
            )
        return np.where(daylight, prediction, 0.0)


def _daylight_mask(features: pd.DataFrame) -> np.ndarray:
    return features["solar_cos_zenith"].to_numpy(dtype=float) > 0.0


def _predict_models(
    models: dict[str, object],
    features: pd.DataFrame,
) -> dict[str, np.ndarray]:
    selected_features = features.loc[:, list(FEATURE_COLUMNS)]
    daylight = _daylight_mask(features)
    return {
        name: np.where(daylight, np.maximum(estimator.predict(selected_features), 0.0), 0.0)
        for name, estimator in models.items()
    }


def fit_and_evaluate(frame: pd.DataFrame, random_state: int = 42) -> ModelAudit:
    """Fit on the earliest 60%; calibrate, choose and test on later periods."""
    history = normalize_hourly_frame(frame)
    if TARGET not in history:
        raise ValueError(f"historical data must include '{TARGET}'")
    features, target = prepare_supervised_frame(history)
    persistence = (
        pd.to_numeric(history[TARGET], errors="coerce")
        .shift(24)
        .reindex(target.index)
        .to_numpy(dtype=float)
    )
    valid_persistence = np.isfinite(persistence)
    if not valid_persistence.all():
        features = features.loc[valid_persistence]
        target = target.loc[valid_persistence]
        persistence = persistence[valid_persistence]
    if len(target) < 500:
        raise ValueError(
            f"only {len(target)} usable hourly records; at least 500 are required"
        )

    partitions = chronological_partitions(len(target))
    train_slice = partitions["train"]
    calibration_slice = partitions["calibration"]
    validation_slice = partitions["validation"]
    test_slice = partitions["test"]

    x_train = features.iloc[train_slice]
    y_train = target.iloc[train_slice].to_numpy(dtype=float)
    x_calibration = features.iloc[calibration_slice]
    y_calibration = target.iloc[calibration_slice].to_numpy(dtype=float)
    x_validation = features.iloc[validation_slice]
    y_validation = target.iloc[validation_slice].to_numpy(dtype=float)
    x_test = features.iloc[test_slice]
    y_test = target.iloc[test_slice]
    persistence_validation = persistence[validation_slice]
    persistence_test = persistence[test_slice]

    models: dict[str, object] = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_features=0.85,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        ),
    }
    for estimator in models.values():
        estimator.fit(x_train.loc[:, list(FEATURE_COLUMNS)], y_train)

    validation_predictions = _predict_models(models, x_validation)
    validation_daylight = _daylight_mask(x_validation)
    validation_predictions[NWP_NAME] = np.where(
        validation_daylight,
        np.maximum(x_validation["nwp_ghi"].to_numpy(dtype=float), 0.0),
        0.0,
    )
    validation_predictions[PERSISTENCE_NAME] = np.where(
        validation_daylight, np.maximum(persistence_validation, 0.0), 0.0
    )
    validation_metrics = {
        name: score_overall_and_daylight(
            y_validation, prediction, validation_daylight
        )
        for name, prediction in validation_predictions.items()
    }
    model_selection_names = (*MODEL_NAMES, NWP_NAME)
    selected_source = min(
        model_selection_names,
        key=lambda name: validation_metrics[name]["daylight"]["mae"],
    )

    calibration_predictions = _predict_models(models, x_calibration)
    calibration_daylight = _daylight_mask(x_calibration)
    calibration_predictions[NWP_NAME] = np.where(
        calibration_daylight,
        np.maximum(x_calibration["nwp_ghi"].to_numpy(dtype=float), 0.0),
        0.0,
    )
    calibration_predictions[PERSISTENCE_NAME] = np.where(
        calibration_daylight,
        np.maximum(persistence[calibration_slice], 0.0),
        0.0,
    )
    interval_residuals = empirical_residual_interval(
        y_calibration[calibration_daylight],
        calibration_predictions[selected_source][calibration_daylight],
        coverage=0.80,
    )

    test_predictions = _predict_models(models, x_test)
    test_daylight = _daylight_mask(x_test)
    test_predictions[NWP_NAME] = np.where(
        test_daylight,
        np.maximum(x_test["nwp_ghi"].to_numpy(dtype=float), 0.0),
        0.0,
    )
    test_predictions[PERSISTENCE_NAME] = np.where(
        test_daylight, np.maximum(persistence_test, 0.0), 0.0
    )
    test_metrics = {
        name: score_overall_and_daylight(y_test, prediction, test_daylight)
        for name, prediction in test_predictions.items()
    }

    interval_test_coverage: float | None = None
    interval_reliable = False
    if interval_residuals is not None and len(y_test) >= 100:
        residual_low, residual_high = interval_residuals
        daylight_actual = y_test.to_numpy(dtype=float)[test_daylight]
        lower = np.maximum(
            0.0, test_predictions[selected_source][test_daylight] + residual_low
        )
        upper = np.maximum(
            lower,
            test_predictions[selected_source][test_daylight] + residual_high,
        )
        interval_test_coverage = interval_coverage(daylight_actual, lower, upper)
        interval_reliable = 0.65 <= interval_test_coverage <= 0.95

    explained_model = min(
        MODEL_NAMES,
        key=lambda name: validation_metrics[name]["daylight"]["mae"],
    )
    importance_daylight = calibration_daylight
    importance_result = permutation_importance(
        models[explained_model],
        x_calibration.loc[importance_daylight, list(FEATURE_COLUMNS)],
        y_calibration[importance_daylight],
        n_repeats=3,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    feature_importance = pd.Series(
        importance_result.importances_mean,
        index=FEATURE_COLUMNS,
        name="MAE increase (W/m²)",
    ).sort_values(ascending=False)

    return ModelAudit(
        models=models,
        feature_names=FEATURE_COLUMNS,
        selected_source=selected_source,
        explained_model=explained_model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        actual_test=y_test,
        test_features=x_test.copy(),
        predictions_test=pd.DataFrame(test_predictions, index=y_test.index),
        interval_residuals=interval_residuals,
        interval_test_coverage=interval_test_coverage,
        interval_reliable=interval_reliable,
        feature_importance=feature_importance,
        split_dates={
            name: f"{target.index[part.start].isoformat()} — "
            f"{target.index[part.stop - 1].isoformat()}"
            for name, part in partitions.items()
        },
        sample_counts={name: part.stop - part.start for name, part in partitions.items()},
    )

