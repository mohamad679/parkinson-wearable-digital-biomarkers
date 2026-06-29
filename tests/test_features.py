import math

import pytest

from parkinson_wearable_biomarkers.data import DataSchema
from parkinson_wearable_biomarkers.features import FeatureExtractionError, extract_features
from parkinson_wearable_biomarkers.preprocessing import WindowedDataset

SCHEMA = DataSchema(
    accelerometer_columns=("acc_x", "acc_y", "acc_z"),
    subject_id_column="subject",
    label_column="label",
)


def _make_windowed(windows, subject_ids=None, labels=None, window_size=None):
    normalized_windows = tuple(
        tuple(tuple(float(value) for value in sample) for sample in window)
        for window in windows
    )
    size = window_size if window_size is not None else len(normalized_windows[0])
    count = len(normalized_windows)
    return WindowedDataset(
        schema=SCHEMA,
        windows=normalized_windows,
        subject_ids=tuple(subject_ids or ["s1"] * count),
        labels=tuple(labels or [0] * count),
        window_size=size,
    )


def _feature_map(feature_dataset, row=0):
    return dict(zip(feature_dataset.feature_names, feature_dataset.features[row], strict=True))


def test_extract_features_has_stable_shape_names_and_metadata():
    window = ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
    dataset = _make_windowed(
        [window, window],
        subject_ids=["s1", "s2"],
        labels=[0, 1],
    )

    extracted = extract_features(dataset, sampling_rate_hz=4, rolling_window_size=2)

    expected_names = tuple(
        f"{signal}__{feature}"
        for signal in ("acc_x", "acc_y", "acc_z", "magnitude")
        for feature in (
            "mean",
            "std",
            "min",
            "max",
            "energy",
            "dominant_frequency_hz",
        )
    ) + ("magnitude__rolling_variance_mean_w2",)
    assert extracted.shape == (2, 25)
    assert extracted.feature_names == expected_names
    assert extracted.subject_ids == ("s1", "s2")
    assert extracted.labels == (0, 1)


def test_extract_features_computes_statistics_energy_and_rolling_variance():
    dataset = _make_windowed(
        [((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))]
    )

    extracted = extract_features(dataset, sampling_rate_hz=4, rolling_window_size=2)
    features = _feature_map(extracted)

    assert features["acc_x__mean"] == pytest.approx(1.5)
    assert features["acc_x__std"] == pytest.approx(math.sqrt(1.25))
    assert features["acc_x__min"] == 0.0
    assert features["acc_x__max"] == 3.0
    assert features["acc_x__energy"] == 14.0
    assert features["magnitude__mean"] == pytest.approx(1.5)
    assert features["magnitude__rolling_variance_mean_w2"] == pytest.approx(0.25)


def test_extract_features_uses_euclidean_acceleration_magnitude():
    dataset = _make_windowed(
        [((3, 4, 0), (-3, 4, 0), (0, 0, 0), (0, 0, 0))]
    )

    features = _feature_map(extract_features(dataset, sampling_rate_hz=4))

    assert features["magnitude__mean"] == 2.5
    assert features["magnitude__max"] == 5.0
    assert features["magnitude__energy"] == 50.0


def test_extract_features_finds_axis_and_magnitude_dominant_frequency():
    dataset = _make_windowed(
        [((0, 0, 0), (1, 0, 0), (0, 0, 0), (-1, 0, 0))]
    )

    features = _feature_map(extract_features(dataset, sampling_rate_hz=4))

    assert features["acc_x__dominant_frequency_hz"] == pytest.approx(1.0)
    assert features["magnitude__dominant_frequency_hz"] == pytest.approx(2.0)


def test_extract_features_handles_partial_and_all_nan_signals():
    nan = math.nan
    dataset = _make_windowed(
        [
            ((nan, 1, 0), (2, nan, 0), (4, 3, nan), (nan, nan, nan)),
            ((nan, nan, nan),) * 4,
        ]
    )

    extracted = extract_features(dataset, sampling_rate_hz=100)
    first = _feature_map(extracted, row=0)

    assert first["acc_x__mean"] == 3.0
    assert first["acc_y__mean"] == 2.0
    assert first["acc_z__mean"] == 0.0
    assert all(
        value == 0.0
        for name, value in first.items()
        if name.startswith("magnitude__")
    )
    assert all(math.isfinite(value) for row in extracted.features for value in row)
    assert all(value == 0.0 for value in extracted.features[1])


def test_extract_features_is_deterministic():
    dataset = _make_windowed(
        [((0.5, 1, 2), (1.5, 2, 3), (0.5, 3, 4), (-0.5, 4, 5))]
    )

    first = extract_features(dataset, sampling_rate_hz=50, rolling_window_size=3)
    second = extract_features(dataset, sampling_rate_hz=50, rolling_window_size=3)

    assert first == second


def test_extract_features_supports_empty_window_collection():
    dataset = _make_windowed([], window_size=4)

    extracted = extract_features(dataset, sampling_rate_hz=100)

    assert extracted.shape == (0, 25)
    assert extracted.features == ()


@pytest.mark.parametrize("sampling_rate_hz", [0, -1, math.nan, math.inf, True, "100"])
def test_extract_features_rejects_invalid_sampling_rate(sampling_rate_hz):
    dataset = _make_windowed([((0, 0, 0),) * 4])

    with pytest.raises(FeatureExtractionError, match="Sampling rate"):
        extract_features(dataset, sampling_rate_hz=sampling_rate_hz)


@pytest.mark.parametrize("rolling_window_size", [0, 5, 1.5, True])
def test_extract_features_rejects_invalid_rolling_window_size(rolling_window_size):
    dataset = _make_windowed([((0, 0, 0),) * 4])

    with pytest.raises(FeatureExtractionError, match="Rolling window size"):
        extract_features(
            dataset,
            sampling_rate_hz=100,
            rolling_window_size=rolling_window_size,
        )
