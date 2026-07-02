import math

import pytest

from parkinson_wearable_biomarkers.calibration import (
    CalibrationInputError,
    brier_score,
    calibration_curve_data,
    expected_calibration_error,
)


def test_brier_score_on_toy_data():
    assert brier_score((0, 1), (0.25, 0.75)) == pytest.approx(0.0625)


def test_calibration_curve_and_expected_calibration_error():
    labels = (0, 0, 1, 1)
    probabilities = (0.1, 0.4, 0.6, 0.9)

    curve = calibration_curve_data(labels, probabilities, n_bins=2)

    assert len(curve) == 2
    assert curve[0].bin_index == 0
    assert curve[0].count == 2
    assert curve[0].mean_probability == pytest.approx(0.25)
    assert curve[0].observed_positive_rate == 0.0
    assert curve[1].bin_index == 1
    assert curve[1].count == 2
    assert curve[1].mean_probability == pytest.approx(0.75)
    assert curve[1].observed_positive_rate == 1.0
    assert expected_calibration_error(labels, probabilities, n_bins=2) == pytest.approx(0.25)


def test_calibration_bins_are_left_closed_and_include_probability_one():
    curve = calibration_curve_data((0, 1, 1), (0.0, 0.5, 1.0), n_bins=2)

    assert [(item.bin_index, item.count) for item in curve] == [(0, 1), (1, 2)]
    assert curve[0].lower_bound == 0.0
    assert curve[0].upper_bound == 0.5
    assert curve[1].lower_bound == 0.5
    assert curve[1].upper_bound == 1.0


def test_perfect_probability_estimates_have_zero_calibration_error():
    labels = (0, 0, 1, 1)
    probabilities = (0.0, 0.0, 1.0, 1.0)

    assert brier_score(labels, probabilities) == 0.0
    assert expected_calibration_error(labels, probabilities, n_bins=10) == 0.0


def test_calibration_functions_are_deterministic():
    labels = (0, 1, 0, 1)
    probabilities = (0.2, 0.7, 0.4, 0.9)

    first = calibration_curve_data(labels, probabilities, n_bins=4)
    second = calibration_curve_data(labels, probabilities, n_bins=4)

    assert first == second
    assert expected_calibration_error(
        labels, probabilities, n_bins=4
    ) == expected_calibration_error(labels, probabilities, n_bins=4)


@pytest.mark.parametrize(
    ("labels", "probabilities", "message"),
    [
        ((), (), "must not be empty"),
        ((0, 1), (0.2,), "equal lengths"),
        ((0, 2), (0.2, 0.8), "Labels"),
        ((False, 1), (0.2, 0.8), "Labels"),
        ((0, 1), (-0.1, 0.8), "Probabilities"),
        ((0, 1), (0.2, 1.1), "Probabilities"),
        ((0, 1), (0.2, math.nan), "Probabilities"),
        ((0, 1), (0.2, True), "Probabilities"),
        (None, (0.2, 0.8), "Labels"),
        ((0, 1), None, "Probabilities"),
    ],
)
def test_calibration_rejects_invalid_inputs(labels, probabilities, message):
    with pytest.raises(CalibrationInputError, match=message):
        brier_score(labels, probabilities)


@pytest.mark.parametrize("n_bins", [0, -1, 1.5, True])
def test_calibration_rejects_invalid_bin_count(n_bins):
    with pytest.raises(CalibrationInputError, match="calibration bins"):
        calibration_curve_data((0, 1), (0.2, 0.8), n_bins=n_bins)
