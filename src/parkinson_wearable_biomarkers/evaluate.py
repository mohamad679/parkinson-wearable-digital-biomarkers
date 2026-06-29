"""Binary discrimination and threshold metrics for imbalanced FoG evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real


class MetricInputError(ValueError):
    """Raised when binary metric inputs are invalid or a metric is undefined."""


@dataclass(frozen=True, slots=True)
class ConfusionMatrix:
    """Binary confusion counts arranged as ``((TN, FP), (FN, TP))``."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def __post_init__(self) -> None:
        counts = (
            self.true_negative,
            self.false_positive,
            self.false_negative,
            self.true_positive,
        )
        if any(isinstance(count, bool) or not isinstance(count, Integral) for count in counts):
            raise MetricInputError("Confusion matrix counts must be integers")
        if any(count < 0 for count in counts):
            raise MetricInputError("Confusion matrix counts must be non-negative")

    @property
    def total(self) -> int:
        """Return the total number of observations."""
        return self.true_negative + self.false_positive + self.false_negative + self.true_positive

    @property
    def positive_support(self) -> int:
        """Return the number of observed positive labels."""
        return self.true_positive + self.false_negative

    @property
    def negative_support(self) -> int:
        """Return the number of observed negative labels."""
        return self.true_negative + self.false_positive

    @property
    def precision(self) -> float:
        """Return positive predictive value, or zero with no positive predictions."""
        predicted_positive = self.true_positive + self.false_positive
        return self.true_positive / predicted_positive if predicted_positive else 0.0

    @property
    def f1(self) -> float:
        """Return the positive-class F1 score."""
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 0.0

    @property
    def sensitivity(self) -> float:
        """Return true-positive rate."""
        if not self.positive_support:
            raise MetricInputError("Sensitivity requires at least one positive label")
        return self.true_positive / self.positive_support

    @property
    def specificity(self) -> float:
        """Return true-negative rate."""
        if not self.negative_support:
            raise MetricInputError("Specificity requires at least one negative label")
        return self.true_negative / self.negative_support

    @property
    def positive_prevalence(self) -> float:
        """Return observed positive prevalence for imbalance context."""
        if not self.total:
            raise MetricInputError("Prevalence requires at least one observation")
        return self.positive_support / self.total

    def as_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return counts in conventional matrix row order."""
        return (
            (self.true_negative, self.false_positive),
            (self.false_negative, self.true_positive),
        )


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Positive-class metrics at one probability threshold."""

    threshold: float
    confusion_matrix: ConfusionMatrix

    @property
    def precision(self) -> float:
        return self.confusion_matrix.precision

    @property
    def f1(self) -> float:
        return self.confusion_matrix.f1

    @property
    def sensitivity(self) -> float:
        return self.confusion_matrix.sensitivity

    @property
    def specificity(self) -> float:
        return self.confusion_matrix.specificity

    @property
    def predicted_positive_rate(self) -> float:
        predicted_positive = (
            self.confusion_matrix.true_positive + self.confusion_matrix.false_positive
        )
        return predicted_positive / self.confusion_matrix.total


def auroc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Compute tie-aware area under the ROC curve using average ranks."""
    normalized_labels, normalized_probabilities = _validate_binary_inputs(
        labels, probabilities
    )
    positive_count, negative_count = _require_both_classes(normalized_labels, "AUROC")

    ordered = sorted(
        zip(normalized_probabilities, normalized_labels, strict=True),
        key=lambda item: item[0],
    )
    positive_rank_sum = 0.0
    group_start = 0
    while group_start < len(ordered):
        group_end = group_start + 1
        while group_end < len(ordered) and ordered[group_end][0] == ordered[group_start][0]:
            group_end += 1

        average_rank = ((group_start + 1) + group_end) / 2.0
        positives_in_group = sum(label for _, label in ordered[group_start:group_end])
        positive_rank_sum += positives_in_group * average_rank
        group_start = group_end

    minimum_positive_rank_sum = positive_count * (positive_count + 1) / 2.0
    return (positive_rank_sum - minimum_positive_rank_sum) / (
        positive_count * negative_count
    )


def auprc(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Compute non-interpolated average precision with score ties grouped."""
    normalized_labels, normalized_probabilities = _validate_binary_inputs(
        labels, probabilities
    )
    positive_count, _ = _require_both_classes(normalized_labels, "AUPRC")

    ordered = sorted(
        zip(normalized_probabilities, normalized_labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positive = 0
    false_positive = 0
    average_precision = 0.0
    group_start = 0
    while group_start < len(ordered):
        group_end = group_start + 1
        while group_end < len(ordered) and ordered[group_end][0] == ordered[group_start][0]:
            group_end += 1

        positives_in_group = sum(label for _, label in ordered[group_start:group_end])
        negatives_in_group = group_end - group_start - positives_in_group
        true_positive += positives_in_group
        false_positive += negatives_in_group
        if positives_in_group:
            recall_increment = positives_in_group / positive_count
            precision = true_positive / (true_positive + false_positive)
            average_precision += recall_increment * precision
        group_start = group_end

    return average_precision


def confusion_matrix(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> ConfusionMatrix:
    """Compute binary confusion counts using ``probability >= threshold``."""
    normalized_labels, normalized_probabilities = _validate_binary_inputs(
        labels, probabilities
    )
    normalized_threshold = _validate_threshold(threshold)
    return _confusion_matrix(normalized_labels, normalized_probabilities, normalized_threshold)


def f1_score(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> float:
    """Compute positive-class F1 at ``threshold``."""
    return confusion_matrix(labels, probabilities, threshold=threshold).f1


def sensitivity(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> float:
    """Compute true-positive rate at ``threshold``."""
    return confusion_matrix(labels, probabilities, threshold=threshold).sensitivity


def specificity(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float = 0.5
) -> float:
    """Compute true-negative rate at ``threshold``."""
    return confusion_matrix(labels, probabilities, threshold=threshold).specificity


def threshold_analysis(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    thresholds: Sequence[float],
) -> tuple[ThresholdMetrics, ...]:
    """Compute imbalance-aware metrics for each supplied threshold in order."""
    normalized_labels, normalized_probabilities = _validate_binary_inputs(
        labels, probabilities
    )
    _require_both_classes(normalized_labels, "Threshold analysis")
    normalized_thresholds = _validate_thresholds(thresholds)

    return tuple(
        ThresholdMetrics(
            threshold=threshold,
            confusion_matrix=_confusion_matrix(
                normalized_labels, normalized_probabilities, threshold
            ),
        )
        for threshold in normalized_thresholds
    )


def _confusion_matrix(
    labels: tuple[int, ...],
    probabilities: tuple[float, ...],
    threshold: float,
) -> ConfusionMatrix:
    true_negative = false_positive = false_negative = true_positive = 0
    for label, probability in zip(labels, probabilities, strict=True):
        prediction = int(probability >= threshold)
        if label == 1 and prediction == 1:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif prediction == 1:
            false_positive += 1
        else:
            true_negative += 1

    return ConfusionMatrix(
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def _validate_binary_inputs(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    normalized_labels = _as_tuple(labels, "Labels")
    normalized_probabilities = _as_tuple(probabilities, "Probabilities")
    if not normalized_labels:
        raise MetricInputError("Metric inputs must not be empty")
    if len(normalized_labels) != len(normalized_probabilities):
        raise MetricInputError("Labels and probabilities must have equal lengths")

    for label in normalized_labels:
        if isinstance(label, bool) or not isinstance(label, Integral) or int(label) not in {0, 1}:
            raise MetricInputError("Labels must contain only integer values 0 and 1")

    validated_probabilities: list[float] = []
    for probability in normalized_probabilities:
        if isinstance(probability, bool) or not isinstance(probability, Real):
            raise MetricInputError("Probabilities must be real numbers")
        normalized_probability = float(probability)
        if not math.isfinite(normalized_probability) or not 0.0 <= normalized_probability <= 1.0:
            raise MetricInputError("Probabilities must be finite values between 0 and 1")
        validated_probabilities.append(normalized_probability)

    return tuple(int(label) for label in normalized_labels), tuple(validated_probabilities)


def _require_both_classes(labels: tuple[int, ...], metric_name: str) -> tuple[int, int]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise MetricInputError(f"{metric_name} requires both binary classes")
    return positive_count, negative_count


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise MetricInputError("Thresholds must be real numbers")
    normalized_threshold = float(threshold)
    if not math.isfinite(normalized_threshold) or not 0.0 <= normalized_threshold <= 1.0:
        raise MetricInputError("Thresholds must be finite values between 0 and 1")
    return normalized_threshold


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[float, ...]:
    supplied_thresholds = _as_tuple(thresholds, "Thresholds")
    if not supplied_thresholds:
        raise MetricInputError("At least one threshold is required")
    normalized = tuple(_validate_threshold(threshold) for threshold in supplied_thresholds)
    if len(set(normalized)) != len(normalized):
        raise MetricInputError("Thresholds must be unique")
    return normalized


def _as_tuple(values: Sequence[object], input_name: str) -> tuple[object, ...]:
    try:
        return tuple(values)
    except TypeError as error:
        raise MetricInputError(f"{input_name} must be a sequence") from error
