import math

import pytest

from parkinson_wearable_biomarkers.evaluate import (
    MetricInputError,
    auprc,
    auroc,
    confusion_matrix,
    f1_score,
    sensitivity,
    specificity,
    threshold_analysis,
)

LABELS = (0, 0, 1, 1)
PROBABILITIES = (0.1, 0.4, 0.35, 0.8)


def test_discrimination_metrics_on_toy_data():
    assert auroc(LABELS, PROBABILITIES) == pytest.approx(0.75)
    assert auprc(LABELS, PROBABILITIES) == pytest.approx(5 / 6)


def test_discrimination_metrics_group_tied_scores_deterministically():
    labels = (0, 1)
    probabilities = (0.5, 0.5)

    assert auroc(labels, probabilities) == 0.5
    assert auprc(labels, probabilities) == 0.5


def test_confusion_f1_sensitivity_and_specificity():
    matrix = confusion_matrix(LABELS, PROBABILITIES, threshold=0.5)

    assert matrix.as_rows() == ((2, 0), (1, 1))
    assert matrix.positive_support == 2
    assert matrix.negative_support == 2
    assert matrix.positive_prevalence == 0.5
    assert matrix.precision == 1.0
    assert f1_score(LABELS, PROBABILITIES, threshold=0.5) == pytest.approx(2 / 3)
    assert sensitivity(LABELS, PROBABILITIES, threshold=0.5) == 0.5
    assert specificity(LABELS, PROBABILITIES, threshold=0.5) == 1.0


def test_threshold_analysis_uses_configured_thresholds_in_order():
    results = threshold_analysis(
        LABELS,
        PROBABILITIES,
        thresholds=(0.3, 0.5, 0.9),
    )

    assert [result.threshold for result in results] == [0.3, 0.5, 0.9]
    assert [result.confusion_matrix.as_rows() for result in results] == [
        ((1, 1), (0, 2)),
        ((2, 0), (1, 1)),
        ((2, 0), (2, 0)),
    ]
    assert results[0].f1 == pytest.approx(0.8)
    assert results[0].sensitivity == 1.0
    assert results[0].specificity == 0.5
    assert results[2].f1 == 0.0


def test_auprc_and_support_capture_imbalanced_positive_class():
    labels = (0,) * 9 + (1,)
    probabilities = (0.1,) * 9 + (0.9,)

    assert auroc(labels, probabilities) == 1.0
    assert auprc(labels, probabilities) == 1.0
    result = threshold_analysis(labels, probabilities, thresholds=(0.5,))[0]
    assert result.confusion_matrix.positive_support == 1
    assert result.confusion_matrix.negative_support == 9
    assert result.confusion_matrix.positive_prevalence == 0.1


def test_metrics_are_deterministic():
    first = threshold_analysis(LABELS, PROBABILITIES, thresholds=(0.25, 0.5, 0.75))
    second = threshold_analysis(LABELS, PROBABILITIES, thresholds=(0.25, 0.5, 0.75))

    assert first == second
    assert auroc(LABELS, PROBABILITIES) == auroc(LABELS, PROBABILITIES)
    assert auprc(LABELS, PROBABILITIES) == auprc(LABELS, PROBABILITIES)


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
def test_metrics_reject_invalid_inputs(labels, probabilities, message):
    with pytest.raises(MetricInputError, match=message):
        auroc(labels, probabilities)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.5"])
def test_confusion_matrix_rejects_invalid_thresholds(threshold):
    with pytest.raises(MetricInputError, match="Thresholds"):
        confusion_matrix(LABELS, PROBABILITIES, threshold=threshold)


@pytest.mark.parametrize("thresholds", [(), (0.5, 0.5), 0.5])
def test_threshold_analysis_rejects_empty_or_duplicate_thresholds(thresholds):
    with pytest.raises(MetricInputError, match="[Tt]hreshold"):
        threshold_analysis(LABELS, PROBABILITIES, thresholds=thresholds)


@pytest.mark.parametrize("metric", [auroc, auprc])
def test_discrimination_metrics_require_both_classes(metric):
    with pytest.raises(MetricInputError, match="requires both binary classes"):
        metric((0, 0), (0.1, 0.2))
