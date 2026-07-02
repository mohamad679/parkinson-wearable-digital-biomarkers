import pytest

from parkinson_wearable_biomarkers.features import FeatureDataset
from parkinson_wearable_biomarkers.validation import (
    SubjectLeakageError,
    SubjectSplitError,
    assert_no_subject_leakage,
    group_k_fold_splits,
    leave_one_subject_out_splits,
)


def _make_dataset(subject_ids):
    return FeatureDataset(
        feature_names=("feature_a", "feature_b"),
        features=tuple(
            (float(index), float(index % 2)) for index in range(len(subject_ids))
        ),
        subject_ids=tuple(subject_ids),
        labels=tuple(index % 2 for index in range(len(subject_ids))),
    )


def _subjects(dataset, indices):
    return {dataset.subject_ids[index] for index in indices}


def test_group_k_fold_has_no_subject_leakage_and_complete_test_coverage():
    dataset = _make_dataset(
        ["s1"] * 4 + ["s2"] * 3 + ["s3"] * 2 + ["s4"]
    )

    folds = group_k_fold_splits(dataset, n_splits=2)

    assert len(folds) == 2
    assert [len(fold.test_indices) for fold in folds] == [5, 5]
    assert sorted(index for fold in folds for index in fold.test_indices) == list(
        range(len(dataset))
    )
    for fold in folds:
        assert_no_subject_leakage(dataset, fold.train_indices, fold.test_indices)
        assert _subjects(dataset, fold.train_indices).isdisjoint(
            _subjects(dataset, fold.test_indices)
        )


@pytest.mark.parametrize("n_splits", [0, 1, -1, 2.5, True])
def test_group_k_fold_rejects_invalid_number_of_folds(n_splits):
    dataset = _make_dataset(["s1", "s2", "s3"])

    with pytest.raises(SubjectSplitError, match="Number of folds"):
        group_k_fold_splits(dataset, n_splits=n_splits)


def test_group_k_fold_rejects_more_folds_than_subjects():
    dataset = _make_dataset(["s1", "s1", "s2", "s2"])

    with pytest.raises(SubjectSplitError, match="cannot exceed unique subjects"):
        group_k_fold_splits(dataset, n_splits=3)


def test_subject_aware_splitting_rejects_a_single_subject():
    dataset = _make_dataset(["s1", "s1", "s1"])

    with pytest.raises(SubjectSplitError, match="at least 2 unique subjects"):
        group_k_fold_splits(dataset, n_splits=2)
    with pytest.raises(SubjectSplitError, match="at least 2 unique subjects"):
        leave_one_subject_out_splits(dataset)


def test_group_k_fold_is_deterministic():
    dataset = _make_dataset(
        ["s3", "s1", "s2", "s1", "s3", "s4", "s2", "s3"]
    )

    first = group_k_fold_splits(dataset, n_splits=3)
    second = group_k_fold_splits(dataset, n_splits=3)

    assert first == second


def test_leave_one_subject_out_uses_first_appearance_order():
    dataset = _make_dataset(["s2", "s2", "s1", "s3", "s1", "s3"])

    folds = leave_one_subject_out_splits(dataset)

    assert [fold.fold_index for fold in folds] == [0, 1, 2]
    assert [fold.test_indices for fold in folds] == [(0, 1), (2, 4), (3, 5)]
    assert [
        _subjects(dataset, fold.test_indices) for fold in folds
    ] == [{"s2"}, {"s1"}, {"s3"}]
    for fold in folds:
        assert_no_subject_leakage(dataset, fold.train_indices, fold.test_indices)


def test_explicit_leakage_check_rejects_shared_subjects():
    dataset = _make_dataset(["s1", "s1", "s2", "s2"])

    with pytest.raises(SubjectLeakageError, match="s1"):
        assert_no_subject_leakage(dataset, train_indices=(0, 2), test_indices=(1, 3))
