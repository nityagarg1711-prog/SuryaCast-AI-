"""Chronological splits and consistent regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def chronological_partitions(
    sample_count: int,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.10,
) -> dict[str, slice]:
    """Partition ordered rows into train/calibration/validation/test segments."""
    if sample_count < 20:
        raise ValueError("at least 20 chronological samples are required")
    fractions = (train_fraction, calibration_fraction, validation_fraction)
    if any(fraction <= 0 for fraction in fractions) or sum(fractions) >= 1:
        raise ValueError("split fractions must be positive and leave a test segment")
    train_end = int(sample_count * train_fraction)
    calibration_end = train_end + int(sample_count * calibration_fraction)
    validation_end = calibration_end + int(sample_count * validation_fraction)
    if train_end < 10 or calibration_end <= train_end or validation_end <= calibration_end:
        raise ValueError("not enough samples for the requested chronological splits")
    return {
        "train": slice(0, train_end),
        "calibration": slice(train_end, calibration_end),
        "validation": slice(calibration_end, validation_end),
        "test": slice(validation_end, sample_count),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute MAE, RMSE and R² in the original irradiance units (W/m²)."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted values must have equal, non-empty shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite values")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def score_overall_and_daylight(
    y_true,
    y_pred,
    daylight_mask,
) -> dict[str, dict[str, float]]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    daylight = np.asarray(daylight_mask, dtype=bool).reshape(-1)
    if actual.shape != predicted.shape or actual.shape != daylight.shape:
        raise ValueError("actual, predictions, and daylight mask must align")
    if not daylight.any():
        raise ValueError("daylight metrics require at least one daylight sample")
    return {
        "all_hours": regression_metrics(actual, predicted),
        "daylight": regression_metrics(actual[daylight], predicted[daylight]),
    }


def empirical_residual_interval(
    y_true,
    y_pred,
    coverage: float = 0.80,
    minimum_samples: int = 100,
) -> tuple[float, float] | None:
    """Return additive calibration residual quantiles, if sample size permits."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predictions must have equal shapes")
    if actual.size < minimum_samples:
        return None
    residuals = actual - predicted
    tail_probability = (1.0 - coverage) / 2.0
    lower, upper = np.quantile(residuals, [tail_probability, 1.0 - tail_probability])
    return float(lower), float(upper)


def interval_coverage(y_true, lower, upper) -> float:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    low = np.asarray(lower, dtype=float).reshape(-1)
    high = np.asarray(upper, dtype=float).reshape(-1)
    if actual.shape != low.shape or actual.shape != high.shape or actual.size == 0:
        raise ValueError("interval bounds must align with non-empty actual values")
    return float(np.mean((actual >= low) & (actual <= high)))

