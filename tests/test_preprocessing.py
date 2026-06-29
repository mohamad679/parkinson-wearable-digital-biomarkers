import pytest

from parkinson_wearable_biomarkers.data import DataSchema, SensorDataset
from parkinson_wearable_biomarkers.preprocessing import WindowingError, create_windows

SCHEMA = DataSchema(
    accelerometer_columns=("acc_x", "acc_y"),
    subject_id_column="subject",
    label_column="label",
)


def _make_dataset(subject_ids, labels):
    samples = tuple((float(index), float(index) + 0.5) for index in range(len(subject_ids)))
    return SensorDataset(
        schema=SCHEMA,
        samples=samples,
        subject_ids=tuple(subject_ids),
        labels=tuple(labels),
    )


def test_create_windows_has_expected_shape_and_values():
    dataset = _make_dataset(["s1"] * 6, [0] * 6)

    windowed = create_windows(dataset, window_size=3)

    assert windowed.shape == (2, 3, 2)
    assert windowed.windows == (
        ((0.0, 0.5), (1.0, 1.5), (2.0, 2.5)),
        ((3.0, 3.5), (4.0, 4.5), (5.0, 5.5)),
    )


def test_create_windows_applies_overlap_within_subject():
    dataset = _make_dataset(["s1"] * 6, [0] * 6)

    windowed = create_windows(dataset, window_size=4, overlap_fraction=0.5)

    assert windowed.shape == (2, 4, 2)
    assert tuple(sample[0] for sample in windowed.windows[0]) == (0.0, 1.0, 2.0, 3.0)
    assert tuple(sample[0] for sample in windowed.windows[1]) == (2.0, 3.0, 4.0, 5.0)


def test_create_windows_never_crosses_subject_boundaries():
    dataset = _make_dataset(
        ["s1", "s1", "s1", "s2", "s2", "s2"],
        [0, 0, 0, 1, 1, 1],
    )

    windowed = create_windows(dataset, window_size=3, overlap_fraction=0.5)

    assert windowed.subject_ids == ("s1", "s2")
    assert windowed.labels == (0, 1)
    assert tuple(sample[0] for sample in windowed.windows[0]) == (0.0, 1.0, 2.0)
    assert tuple(sample[0] for sample in windowed.windows[1]) == (3.0, 4.0, 5.0)


def test_create_windows_uses_majority_label_and_first_label_for_ties():
    dataset = _make_dataset(
        ["s1"] * 12,
        [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    )

    windowed = create_windows(dataset, window_size=4)

    assert windowed.labels == (1, 1, 0)


def test_create_windows_discards_short_subject_runs_instead_of_combining_them():
    dataset = _make_dataset(
        ["s1", "s1", "s2", "s2", "s1", "s1"],
        [0, 0, 1, 1, 0, 0],
    )

    windowed = create_windows(dataset, window_size=3)

    assert windowed.shape == (0, 3, 2)
    assert windowed.subject_ids == ()
    assert windowed.labels == ()


def test_create_windows_includes_an_exact_size_subject_run():
    dataset = _make_dataset(["s1", "s1", "s1"], [0, 1, 1])

    windowed = create_windows(dataset, window_size=3)

    assert windowed.shape == (1, 3, 2)
    assert windowed.subject_ids == ("s1",)
    assert windowed.labels == (1,)


def test_create_windows_is_deterministic():
    dataset = _make_dataset(["s1"] * 5, [0, 1, 1, 0, 0])

    first = create_windows(dataset, window_size=3, overlap_fraction=0.5)
    second = create_windows(dataset, window_size=3, overlap_fraction=0.5)

    assert first == second


@pytest.mark.parametrize("window_size", [0, -1, 2.5, True])
def test_create_windows_rejects_invalid_window_size(window_size):
    dataset = _make_dataset(["s1"] * 3, [0] * 3)

    with pytest.raises(WindowingError, match="Window size"):
        create_windows(dataset, window_size=window_size)


@pytest.mark.parametrize("overlap_fraction", [-0.1, 1.0, float("nan"), True, "0.5"])
def test_create_windows_rejects_invalid_overlap(overlap_fraction):
    dataset = _make_dataset(["s1"] * 3, [0] * 3)

    with pytest.raises(WindowingError, match="Overlap fraction"):
        create_windows(dataset, window_size=3, overlap_fraction=overlap_fraction)
