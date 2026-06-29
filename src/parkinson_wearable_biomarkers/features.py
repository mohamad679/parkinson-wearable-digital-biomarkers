"""Deterministic signal features for classical machine-learning baselines.

Feature names use ``<signal>__<feature>``. Each configured accelerometer axis is
listed in schema order, followed by the derived ``magnitude`` signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from parkinson_wearable_biomarkers.preprocessing import WindowedDataset


class FeatureExtractionError(ValueError):
    """Raised when feature configuration or output is invalid."""


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    """Finite numeric features with labels and subject groups for model training."""

    feature_names: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    subject_ids: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        row_count = len(self.features)
        if len(self.subject_ids) != row_count or len(self.labels) != row_count:
            raise FeatureExtractionError(
                "Feature rows, subject IDs, and labels must have equal lengths"
            )
        if not self.feature_names:
            raise FeatureExtractionError("At least one feature name is required")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise FeatureExtractionError("Feature names must be unique")

        feature_count = len(self.feature_names)
        if any(len(row) != feature_count for row in self.features):
            raise FeatureExtractionError(f"Every feature row must contain {feature_count} values")
        if any(not math.isfinite(value) for row in self.features for value in row):
            raise FeatureExtractionError("Feature values must be finite")

    def __len__(self) -> int:
        """Return the number of feature rows."""
        return len(self.features)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(window count, feature count)``."""
        return (len(self), len(self.feature_names))


_SIGNAL_FEATURES = (
    "mean",
    "std",
    "min",
    "max",
    "energy",
    "dominant_frequency_hz",
)


def extract_features(
    dataset: WindowedDataset,
    *,
    sampling_rate_hz: float,
    rolling_window_size: int | None = None,
) -> FeatureDataset:
    """Extract a finite feature row for each accelerometer window.

    Statistics ignore non-finite values. Dominant frequency uses a direct DFT,
    excludes the DC component, and mean-imputes non-finite samples. A signal with
    no finite variation yields frequency ``0.0``. Magnitude is available only for
    samples where every configured axis is finite.

    ``rolling_window_size`` defaults to three samples, capped at the input window
    size. Rolling variance is the mean population variance of all consecutive
    magnitude subwindows of that size.
    """
    normalized_sampling_rate = _validate_sampling_rate(sampling_rate_hz)
    normalized_rolling_size = _validate_rolling_window_size(
        rolling_window_size, dataset.window_size
    )
    feature_names = _build_feature_names(
        dataset.schema.accelerometer_columns, normalized_rolling_size
    )
    rows = tuple(
        _extract_window_features(
            window,
            normalized_sampling_rate,
            normalized_rolling_size,
        )
        for window in dataset.windows
    )

    return FeatureDataset(
        feature_names=feature_names,
        features=rows,
        subject_ids=dataset.subject_ids,
        labels=dataset.labels,
    )


def _build_feature_names(
    accelerometer_columns: tuple[str, ...], rolling_window_size: int
) -> tuple[str, ...]:
    names = [
        f"{axis}__{feature}"
        for axis in accelerometer_columns
        for feature in _SIGNAL_FEATURES
    ]
    names.extend(f"magnitude__{feature}" for feature in _SIGNAL_FEATURES)
    names.append(f"magnitude__rolling_variance_mean_w{rolling_window_size}")
    return tuple(names)


def _extract_window_features(
    window: tuple[tuple[float, ...], ...],
    sampling_rate_hz: float,
    rolling_window_size: int,
) -> tuple[float, ...]:
    row: list[float] = []
    axis_count = len(window[0])
    for axis_index in range(axis_count):
        signal = tuple(sample[axis_index] for sample in window)
        row.extend(_signal_features(signal, sampling_rate_hz))

    magnitude = _magnitude_signal(window)
    row.extend(_signal_features(magnitude, sampling_rate_hz))
    row.append(_rolling_variance_mean(magnitude, rolling_window_size))
    return tuple(row)


def _signal_features(values: tuple[float, ...], sampling_rate_hz: float) -> tuple[float, ...]:
    finite_values = tuple(value for value in values if math.isfinite(value))
    if finite_values:
        mean = math.fsum(finite_values) / len(finite_values)
        variance = math.fsum((value - mean) ** 2 for value in finite_values) / len(
            finite_values
        )
        minimum = min(finite_values)
        maximum = max(finite_values)
        energy = math.fsum(value * value for value in finite_values)
    else:
        mean = variance = minimum = maximum = energy = 0.0

    return (
        mean,
        math.sqrt(variance),
        minimum,
        maximum,
        energy,
        _dominant_frequency(values, sampling_rate_hz),
    )


def _magnitude_signal(
    window: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    return tuple(
        math.sqrt(math.fsum(value * value for value in sample))
        if all(math.isfinite(value) for value in sample)
        else math.nan
        for sample in window
    )


def _dominant_frequency(values: tuple[float, ...], sampling_rate_hz: float) -> float:
    if len(values) < 2:
        return 0.0

    finite_values = tuple(value for value in values if math.isfinite(value))
    fill_value = math.fsum(finite_values) / len(finite_values) if finite_values else 0.0
    imputed = tuple(value if math.isfinite(value) else fill_value for value in values)
    signal_mean = math.fsum(imputed) / len(imputed)
    centered = tuple(value - signal_mean for value in imputed)
    if not any(centered):
        return 0.0

    best_bin = 0
    best_power = 0.0
    sample_count = len(centered)
    for frequency_bin in range(1, sample_count // 2 + 1):
        angular_step = 2.0 * math.pi * frequency_bin / sample_count
        real = math.fsum(
            value * math.cos(angular_step * index) for index, value in enumerate(centered)
        )
        imaginary = -math.fsum(
            value * math.sin(angular_step * index) for index, value in enumerate(centered)
        )
        power = real * real + imaginary * imaginary
        if power > best_power:
            best_bin = frequency_bin
            best_power = power

    return best_bin * sampling_rate_hz / sample_count if best_bin else 0.0


def _rolling_variance_mean(values: tuple[float, ...], rolling_window_size: int) -> float:
    variances: list[float] = []
    for start in range(len(values) - rolling_window_size + 1):
        finite_values = tuple(
            value
            for value in values[start : start + rolling_window_size]
            if math.isfinite(value)
        )
        if not finite_values:
            variances.append(0.0)
            continue

        mean = math.fsum(finite_values) / len(finite_values)
        variances.append(
            math.fsum((value - mean) ** 2 for value in finite_values) / len(finite_values)
        )

    return math.fsum(variances) / len(variances) if variances else 0.0


def _validate_sampling_rate(sampling_rate_hz: float) -> float:
    if isinstance(sampling_rate_hz, bool) or not isinstance(sampling_rate_hz, Real):
        raise FeatureExtractionError("Sampling rate must be a real number")

    normalized_rate = float(sampling_rate_hz)
    if not math.isfinite(normalized_rate) or normalized_rate <= 0.0:
        raise FeatureExtractionError("Sampling rate must be finite and greater than zero")
    return normalized_rate


def _validate_rolling_window_size(
    rolling_window_size: int | None, input_window_size: int
) -> int:
    if rolling_window_size is None:
        return min(3, input_window_size)
    if isinstance(rolling_window_size, bool) or not isinstance(rolling_window_size, Integral):
        raise FeatureExtractionError("Rolling window size must be an integer")

    normalized_size = int(rolling_window_size)
    if not 1 <= normalized_size <= input_window_size:
        raise FeatureExtractionError(
            f"Rolling window size must be between 1 and {input_window_size}"
        )
    return normalized_size
