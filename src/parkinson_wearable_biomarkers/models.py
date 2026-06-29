"""Deterministic, class-weighted baseline classifiers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from parkinson_wearable_biomarkers.features import FeatureDataset
from parkinson_wearable_biomarkers.validation import (
    SubjectFold,
    assert_no_subject_leakage,
)

BaselineModelName = Literal["logistic_regression", "random_forest"]
BaselineEstimator = LogisticRegression | RandomForestClassifier


class ModelInputError(ValueError):
    """Raised when model configuration or prediction input is invalid."""


class ModelTrainingError(ModelInputError):
    """Raised when the supplied rows cannot train a binary classifier."""


@dataclass(frozen=True, slots=True)
class TrainedBaselineModel:
    """Fitted estimator bound to an ordered feature schema."""

    model_name: BaselineModelName
    feature_names: tuple[str, ...]
    random_seed: int
    estimator: BaselineEstimator

    def predict_positive_probabilities(
        self,
        dataset: FeatureDataset,
        indices: Sequence[int] | None = None,
    ) -> tuple[float, ...]:
        """Return probabilities for label ``1`` in requested row order."""
        if dataset.feature_names != self.feature_names:
            raise ModelInputError("Prediction feature names must match the training schema")

        prediction_indices = _normalize_indices(
            indices, len(dataset), "Prediction", allow_empty=True
        )
        if not prediction_indices:
            return ()

        feature_rows = [dataset.features[index] for index in prediction_indices]
        probability_rows = self.estimator.predict_proba(feature_rows)
        classes = tuple(int(label) for label in self.estimator.classes_)
        if 1 not in classes:
            raise ModelTrainingError("Fitted estimator does not contain positive class 1")
        positive_column = classes.index(1)
        probabilities = tuple(float(row[positive_column]) for row in probability_rows)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
            raise ModelTrainingError("Estimator produced invalid positive-class probabilities")
        return probabilities


@dataclass(frozen=True, slots=True)
class FoldPredictions:
    """Positive-class probabilities and metadata for one subject-aware test fold."""

    model_name: BaselineModelName
    fold_index: int
    test_indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    labels: tuple[int, ...]
    subject_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        row_count = len(self.test_indices)
        if (
            len(self.probabilities) != row_count
            or len(self.labels) != row_count
            or len(self.subject_ids) != row_count
        ):
            raise ModelInputError("Fold prediction fields must have equal lengths")
        if any(
            not math.isfinite(probability) or not 0.0 <= probability <= 1.0
            for probability in self.probabilities
        ):
            raise ModelInputError("Fold probabilities must be finite values between 0 and 1")


def fit_logistic_regression(
    dataset: FeatureDataset,
    *,
    train_indices: Sequence[int] | None = None,
    random_seed: int = 42,
    max_iterations: int = 1_000,
) -> TrainedBaselineModel:
    """Fit a deterministic class-weighted logistic regression baseline."""
    seed = _validate_random_seed(random_seed)
    iterations = _validate_positive_integer(max_iterations, "Maximum iterations")
    _, features, labels = _training_rows(dataset, train_indices)
    estimator = LogisticRegression(
        l1_ratio=0.0,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
        max_iter=iterations,
    )
    estimator.fit(features, labels)
    _validate_fitted_classes(estimator.classes_)
    return TrainedBaselineModel(
        model_name="logistic_regression",
        feature_names=dataset.feature_names,
        random_seed=seed,
        estimator=estimator,
    )


def fit_random_forest(
    dataset: FeatureDataset,
    *,
    train_indices: Sequence[int] | None = None,
    random_seed: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> TrainedBaselineModel:
    """Fit a deterministic class-weighted random forest baseline."""
    seed = _validate_random_seed(random_seed)
    tree_count = _validate_positive_integer(n_estimators, "Number of estimators")
    normalized_depth = (
        None
        if max_depth is None
        else _validate_positive_integer(max_depth, "Maximum tree depth")
    )
    _, features, labels = _training_rows(dataset, train_indices)
    estimator = RandomForestClassifier(
        n_estimators=tree_count,
        max_depth=normalized_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    estimator.fit(features, labels)
    _validate_fitted_classes(estimator.classes_)
    return TrainedBaselineModel(
        model_name="random_forest",
        feature_names=dataset.feature_names,
        random_seed=seed,
        estimator=estimator,
    )


def fit_predict_fold(
    dataset: FeatureDataset,
    fold: SubjectFold,
    *,
    model_name: BaselineModelName,
    random_seed: int = 42,
    random_forest_estimators: int = 100,
) -> FoldPredictions:
    """Fit one baseline on a subject-aware fold and predict its test rows."""
    assert_no_subject_leakage(dataset, fold.train_indices, fold.test_indices)
    if model_name == "logistic_regression":
        model = fit_logistic_regression(
            dataset,
            train_indices=fold.train_indices,
            random_seed=random_seed,
        )
    elif model_name == "random_forest":
        model = fit_random_forest(
            dataset,
            train_indices=fold.train_indices,
            random_seed=random_seed,
            n_estimators=random_forest_estimators,
        )
    else:
        raise ModelInputError(f"Unsupported baseline model: {model_name!r}")

    probabilities = model.predict_positive_probabilities(dataset, fold.test_indices)
    return FoldPredictions(
        model_name=model_name,
        fold_index=fold.fold_index,
        test_indices=fold.test_indices,
        probabilities=probabilities,
        labels=tuple(dataset.labels[index] for index in fold.test_indices),
        subject_ids=tuple(dataset.subject_ids[index] for index in fold.test_indices),
    )


def _training_rows(
    dataset: FeatureDataset,
    train_indices: Sequence[int] | None,
) -> tuple[tuple[int, ...], list[tuple[float, ...]], list[int]]:
    normalized_indices = _normalize_indices(
        train_indices, len(dataset), "Training", allow_empty=False
    )
    labels = [dataset.labels[index] for index in normalized_indices]
    if set(labels) != {0, 1}:
        raise ModelTrainingError("Training rows must contain both binary labels 0 and 1")
    features = [dataset.features[index] for index in normalized_indices]
    return normalized_indices, features, labels


def _normalize_indices(
    indices: Sequence[int] | None,
    row_count: int,
    split_name: str,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if indices is None:
        normalized = tuple(range(row_count))
    else:
        try:
            supplied_indices = tuple(indices)
        except TypeError as error:
            raise ModelInputError(f"{split_name} indices must be a sequence") from error

        normalized_list: list[int] = []
        for index in supplied_indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise ModelInputError(f"{split_name} indices must be integers")
            normalized_index = int(index)
            if not 0 <= normalized_index < row_count:
                raise ModelInputError(
                    f"{split_name} index {normalized_index} is outside the dataset"
                )
            normalized_list.append(normalized_index)
        normalized = tuple(normalized_list)

    if len(set(normalized)) != len(normalized):
        raise ModelInputError(f"{split_name} indices must be unique")
    if not normalized and not allow_empty:
        raise ModelTrainingError("Training indices must not be empty")
    return normalized


def _validate_random_seed(random_seed: int) -> int:
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise ModelInputError("Random seed must be an integer")
    normalized_seed = int(random_seed)
    if not 0 <= normalized_seed < 2**32:
        raise ModelInputError("Random seed must be between 0 and 2**32 - 1")
    return normalized_seed


def _validate_positive_integer(value: int, parameter_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelInputError(f"{parameter_name} must be an integer")
    normalized_value = int(value)
    if normalized_value <= 0:
        raise ModelInputError(f"{parameter_name} must be greater than zero")
    return normalized_value


def _validate_fitted_classes(classes: Sequence[int]) -> None:
    normalized_classes = {int(label) for label in classes}
    if normalized_classes != {0, 1}:
        raise ModelTrainingError("Fitted estimator must contain binary labels 0 and 1")
