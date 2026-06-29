"""Deterministic calibration metrics for binary probability estimates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real


class CalibrationInputError(ValueError):
    """Raised when calibration labels, probabilities, or bins are invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Summary of one non-empty equal-width probability bin."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_probability: float
    observed_positive_rate: float


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return mean squared probability error for binary outcomes."""
    normalized_labels, normalized_probabilities = _validate_calibration_inputs(
        labels, probabilities
    )
    squared_errors = (
        (probability - label) ** 2
        for label, probability in zip(
            normalized_labels, normalized_probabilities, strict=True
        )
    )
    return math.fsum(squared_errors) / len(normalized_labels)


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Return count-weighted absolute calibration error over equal-width bins."""
    normalized_labels, normalized_probabilities = _validate_calibration_inputs(
        labels, probabilities
    )
    normalized_bin_count = _validate_bin_count(n_bins)
    bins = _build_calibration_bins(
        normalized_labels, normalized_probabilities, normalized_bin_count
    )
    return math.fsum(
        calibration_bin.count
        / len(normalized_labels)
        * abs(
            calibration_bin.mean_probability
            - calibration_bin.observed_positive_rate
        )
        for calibration_bin in bins
    )


def calibration_curve_data(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> tuple[CalibrationBin, ...]:
    """Return non-empty equal-width bins for calibration plotting.

    Bins are ``[lower_bound, upper_bound)`` except the final bin, which includes
    probability ``1.0``.
    """
    normalized_labels, normalized_probabilities = _validate_calibration_inputs(
        labels, probabilities
    )
    normalized_bin_count = _validate_bin_count(n_bins)
    return _build_calibration_bins(
        normalized_labels, normalized_probabilities, normalized_bin_count
    )


def _build_calibration_bins(
    labels: tuple[int, ...],
    probabilities: tuple[float, ...],
    n_bins: int,
) -> tuple[CalibrationBin, ...]:
    bin_probabilities: list[list[float]] = [[] for _ in range(n_bins)]
    bin_labels: list[list[int]] = [[] for _ in range(n_bins)]
    for label, probability in zip(labels, probabilities, strict=True):
        bin_index = min(int(probability * n_bins), n_bins - 1)
        bin_probabilities[bin_index].append(probability)
        bin_labels[bin_index].append(label)

    bins: list[CalibrationBin] = []
    for bin_index, probabilities_in_bin in enumerate(bin_probabilities):
        if not probabilities_in_bin:
            continue
        labels_in_bin = bin_labels[bin_index]
        bins.append(
            CalibrationBin(
                bin_index=bin_index,
                lower_bound=bin_index / n_bins,
                upper_bound=(bin_index + 1) / n_bins,
                count=len(probabilities_in_bin),
                mean_probability=math.fsum(probabilities_in_bin)
                / len(probabilities_in_bin),
                observed_positive_rate=math.fsum(labels_in_bin) / len(labels_in_bin),
            )
        )
    return tuple(bins)


def _validate_calibration_inputs(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    normalized_labels = _as_tuple(labels, "Labels")
    normalized_probabilities = _as_tuple(probabilities, "Probabilities")
    if not normalized_labels:
        raise CalibrationInputError("Calibration inputs must not be empty")
    if len(normalized_labels) != len(normalized_probabilities):
        raise CalibrationInputError("Labels and probabilities must have equal lengths")

    for label in normalized_labels:
        if isinstance(label, bool) or not isinstance(label, Integral) or int(label) not in {0, 1}:
            raise CalibrationInputError("Labels must contain only integer values 0 and 1")

    validated_probabilities: list[float] = []
    for probability in normalized_probabilities:
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise CalibrationInputError("Probabilities must be real numbers")
        normalized_probability = float(probability)
        if not math.isfinite(normalized_probability) or not 0.0 <= normalized_probability <= 1.0:
            raise CalibrationInputError(
                "Probabilities must be finite values between 0 and 1"
            )
        validated_probabilities.append(normalized_probability)

    return tuple(int(label) for label in normalized_labels), tuple(validated_probabilities)


def _validate_bin_count(n_bins: int) -> int:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral):
        raise CalibrationInputError("Number of calibration bins must be an integer")
    normalized_bin_count = int(n_bins)
    if normalized_bin_count <= 0:
        raise CalibrationInputError("Number of calibration bins must be greater than zero")
    return normalized_bin_count


def _as_tuple(values: Sequence[object], input_name: str) -> tuple[object, ...]:
    try:
        return tuple(values)
    except TypeError as error:
        raise CalibrationInputError(f"{input_name} must be a sequence") from error
