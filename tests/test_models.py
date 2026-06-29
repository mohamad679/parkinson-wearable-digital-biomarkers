import pytest

from parkinson_wearable_biomarkers.features import FeatureDataset
from parkinson_wearable_biomarkers.models import (
    ModelInputError,
    ModelTrainingError,
    fit_logistic_regression,
    fit_predict_fold,
    fit_random_forest,
)
from parkinson_wearable_biomarkers.validation import group_k_fold_splits


def _make_dataset():
    subject_ids: list[str] = []
    features: list[tuple[float, float]] = []
    labels: list[int] = []
    for subject_number in range(4):
        subject_id = f"s{subject_number + 1}"
        offset = subject_number * 0.05
        subject_ids.extend([subject_id] * 4)
        features.extend(
            [
                (-2.0 - offset, 0.0),
                (-1.5 - offset, 0.2),
                (1.5 + offset, 0.8),
                (2.0 + offset, 1.0),
            ]
        )
        labels.extend([0, 0, 1, 1])
    return FeatureDataset(
        feature_names=("signal_mean", "signal_energy"),
        features=tuple(features),
        subject_ids=tuple(subject_ids),
        labels=tuple(labels),
    )


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_baseline_models_predict_positive_probability_shape_and_range(model_name):
    dataset = _make_dataset()
    fold = group_k_fold_splits(dataset, n_splits=2)[0]
    if model_name == "logistic_regression":
        model = fit_logistic_regression(
            dataset, train_indices=fold.train_indices, random_seed=7
        )
    else:
        model = fit_random_forest(
            dataset,
            train_indices=fold.train_indices,
            random_seed=7,
            n_estimators=25,
        )

    probabilities = model.predict_positive_probabilities(dataset, fold.test_indices)

    assert len(probabilities) == len(fold.test_indices)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    test_labels = tuple(dataset.labels[index] for index in fold.test_indices)
    positive_probabilities = [
        probability
        for probability, label in zip(probabilities, test_labels, strict=True)
        if label == 1
    ]
    negative_probabilities = [
        probability
        for probability, label in zip(probabilities, test_labels, strict=True)
        if label == 0
    ]
    assert min(positive_probabilities) > max(negative_probabilities)


def test_logistic_regression_is_deterministic_and_class_weighted():
    dataset = _make_dataset()
    fold = group_k_fold_splits(dataset, n_splits=2)[0]

    first = fit_logistic_regression(dataset, train_indices=fold.train_indices, random_seed=11)
    second = fit_logistic_regression(dataset, train_indices=fold.train_indices, random_seed=11)

    assert first.estimator.class_weight == "balanced"
    assert first.estimator.random_state == 11
    assert first.predict_positive_probabilities(
        dataset, fold.test_indices
    ) == second.predict_positive_probabilities(dataset, fold.test_indices)


def test_random_forest_is_deterministic_single_threaded_and_class_weighted():
    dataset = _make_dataset()
    fold = group_k_fold_splits(dataset, n_splits=2)[0]

    first = fit_random_forest(
        dataset, train_indices=fold.train_indices, random_seed=13, n_estimators=25
    )
    second = fit_random_forest(
        dataset, train_indices=fold.train_indices, random_seed=13, n_estimators=25
    )

    assert first.estimator.class_weight == "balanced"
    assert first.estimator.random_state == 13
    assert first.estimator.n_jobs == 1
    assert first.predict_positive_probabilities(
        dataset, fold.test_indices
    ) == second.predict_positive_probabilities(dataset, fold.test_indices)


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_fit_predict_fold_preserves_test_metadata(model_name):
    dataset = _make_dataset()
    fold = group_k_fold_splits(dataset, n_splits=2)[1]

    predictions = fit_predict_fold(
        dataset,
        fold,
        model_name=model_name,
        random_seed=17,
        random_forest_estimators=25,
    )

    assert predictions.model_name == model_name
    assert predictions.fold_index == fold.fold_index
    assert predictions.test_indices == fold.test_indices
    assert len(predictions.probabilities) == len(fold.test_indices)
    assert predictions.labels == tuple(dataset.labels[index] for index in fold.test_indices)
    assert predictions.subject_ids == tuple(
        dataset.subject_ids[index] for index in fold.test_indices
    )


@pytest.mark.parametrize("trainer", [fit_logistic_regression, fit_random_forest])
def test_baseline_training_requires_both_binary_classes(trainer):
    dataset = _make_dataset()

    with pytest.raises(ModelTrainingError, match="both binary labels"):
        trainer(dataset, train_indices=(0, 1))


def test_prediction_rejects_different_feature_schema():
    dataset = _make_dataset()
    model = fit_logistic_regression(dataset, random_seed=19)
    mismatched = FeatureDataset(
        feature_names=("signal_energy", "signal_mean"),
        features=dataset.features,
        subject_ids=dataset.subject_ids,
        labels=dataset.labels,
    )

    with pytest.raises(ModelInputError, match="feature names"):
        model.predict_positive_probabilities(mismatched)


@pytest.mark.parametrize("random_seed", [-1, 2**32, 1.5, True])
def test_baseline_training_rejects_invalid_random_seed(random_seed):
    with pytest.raises(ModelInputError, match="Random seed"):
        fit_logistic_regression(_make_dataset(), random_seed=random_seed)


@pytest.mark.parametrize("n_estimators", [0, -1, 1.5, True])
def test_random_forest_rejects_invalid_estimator_count(n_estimators):
    with pytest.raises(ModelInputError, match="Number of estimators"):
        fit_random_forest(_make_dataset(), n_estimators=n_estimators)


def test_probability_prediction_supports_empty_requested_rows():
    dataset = _make_dataset()
    model = fit_logistic_regression(dataset)

    assert model.predict_positive_probabilities(dataset, indices=()) == ()
