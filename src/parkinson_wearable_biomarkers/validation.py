"""Deterministic subject-aware cross-validation and leakage checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from parkinson_wearable_biomarkers.features import FeatureDataset


class SubjectSplitError(ValueError):
    """Raised when a valid subject-aware split cannot be constructed."""


class SubjectLeakageError(SubjectSplitError):
    """Raised when a subject occurs in both sides of a fold."""


@dataclass(frozen=True, slots=True)
class SubjectFold:
    """Row indices for one subject-aware train/test fold."""

    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise SubjectSplitError("Fold index must be non-negative")
        if not self.train_indices or not self.test_indices:
            raise SubjectSplitError("Each fold must contain training and test rows")
        if len(set(self.train_indices)) != len(self.train_indices):
            raise SubjectSplitError("Training indices must be unique")
        if len(set(self.test_indices)) != len(self.test_indices):
            raise SubjectSplitError("Test indices must be unique")
        if set(self.train_indices).intersection(self.test_indices):
            raise SubjectLeakageError("A row cannot occur in both train and test indices")


def group_k_fold_splits(
    dataset: FeatureDataset,
    *,
    n_splits: int,
) -> tuple[SubjectFold, ...]:
    """Create deterministic GroupKFold-style splits balanced by row count.

    Subjects are ordered by descending row count. Ties retain the order in which
    subjects first appear in ``dataset``. Each complete subject group is assigned
    to the fold with the fewest test rows, with the lowest fold index breaking
    ties. Input rows are never shuffled.
    """
    normalized_splits = _validate_n_splits(n_splits)
    subject_rows = _subject_rows(dataset)
    _validate_subject_count(len(subject_rows), normalized_splits)

    ordered_subjects = sorted(
        enumerate(subject_rows.items()),
        key=lambda item: (-len(item[1][1]), item[0]),
    )
    fold_subjects: list[set[str]] = [set() for _ in range(normalized_splits)]
    fold_sizes = [0] * normalized_splits

    for _, (subject_id, indices) in ordered_subjects:
        fold_index = min(
            range(normalized_splits),
            key=lambda candidate: (fold_sizes[candidate], candidate),
        )
        fold_subjects[fold_index].add(subject_id)
        fold_sizes[fold_index] += len(indices)

    folds = tuple(
        _build_fold(dataset, fold_index, test_subjects)
        for fold_index, test_subjects in enumerate(fold_subjects)
    )
    _validate_test_partition(folds, len(dataset))
    return folds


def leave_one_subject_out_splits(dataset: FeatureDataset) -> tuple[SubjectFold, ...]:
    """Create one deterministic fold per subject in first-appearance order."""
    subject_rows = _subject_rows(dataset)
    _validate_subject_count(len(subject_rows), required_splits=2)

    folds = tuple(
        _build_fold(dataset, fold_index, {subject_id})
        for fold_index, subject_id in enumerate(subject_rows)
    )
    _validate_test_partition(folds, len(dataset))
    return folds


def assert_no_subject_leakage(
    dataset: FeatureDataset,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
) -> None:
    """Raise ``SubjectLeakageError`` if train and test share any subject."""
    normalized_train = _validate_indices(train_indices, len(dataset), "Training")
    normalized_test = _validate_indices(test_indices, len(dataset), "Test")

    if set(normalized_train).intersection(normalized_test):
        raise SubjectLeakageError("A row cannot occur in both train and test indices")

    train_subjects = {dataset.subject_ids[index] for index in normalized_train}
    test_subjects = {dataset.subject_ids[index] for index in normalized_test}
    leaked_subjects = sorted(train_subjects.intersection(test_subjects))
    if leaked_subjects:
        subjects = ", ".join(leaked_subjects)
        raise SubjectLeakageError(f"Subject leakage detected for: {subjects}")


def _subject_rows(dataset: FeatureDataset) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, subject_id in enumerate(dataset.subject_ids):
        grouped.setdefault(subject_id, []).append(index)
    return {subject_id: tuple(indices) for subject_id, indices in grouped.items()}


def _build_fold(
    dataset: FeatureDataset,
    fold_index: int,
    test_subjects: set[str],
) -> SubjectFold:
    test_indices = tuple(
        index
        for index, subject_id in enumerate(dataset.subject_ids)
        if subject_id in test_subjects
    )
    train_indices = tuple(
        index
        for index, subject_id in enumerate(dataset.subject_ids)
        if subject_id not in test_subjects
    )
    assert_no_subject_leakage(dataset, train_indices, test_indices)
    return SubjectFold(
        fold_index=fold_index,
        train_indices=train_indices,
        test_indices=test_indices,
    )


def _validate_n_splits(n_splits: int) -> int:
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral):
        raise SubjectSplitError("Number of folds must be an integer")

    normalized_splits = int(n_splits)
    if normalized_splits < 2:
        raise SubjectSplitError("Number of folds must be at least 2")
    return normalized_splits


def _validate_subject_count(subject_count: int, required_splits: int) -> None:
    if subject_count < 2:
        raise SubjectSplitError("Subject-aware splitting requires at least 2 unique subjects")
    if required_splits > subject_count:
        raise SubjectSplitError(
            f"Number of folds ({required_splits}) cannot exceed unique subjects "
            f"({subject_count})"
        )


def _validate_indices(
    indices: Sequence[int],
    row_count: int,
    split_name: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise SubjectSplitError(f"{split_name} indices must be integers")
        normalized_index = int(index)
        if not 0 <= normalized_index < row_count:
            raise SubjectSplitError(
                f"{split_name} index {normalized_index} is outside the dataset"
            )
        normalized.append(normalized_index)

    if len(set(normalized)) != len(normalized):
        raise SubjectSplitError(f"{split_name} indices must be unique")
    return tuple(normalized)


def _validate_test_partition(folds: tuple[SubjectFold, ...], row_count: int) -> None:
    test_indices = tuple(index for fold in folds for index in fold.test_indices)
    if len(test_indices) != row_count or set(test_indices) != set(range(row_count)):
        raise SubjectSplitError("Every row must occur in exactly one test fold")
