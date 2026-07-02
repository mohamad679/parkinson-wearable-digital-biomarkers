import csv

import pytest

from parkinson_wearable_biomarkers.data import (
    DataSchema,
    DataValidationError,
    load_csv,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_csv_with_configurable_columns(tmp_path):
    csv_path = tmp_path / "synthetic_sensor_data.csv"
    fieldnames = ["participant", "axis_ap", "axis_ml", "axis_v", "fog", "unused"]
    _write_csv(
        csv_path,
        fieldnames,
        [
            {
                "participant": "001",
                "axis_ap": "0.1",
                "axis_ml": "-0.2",
                "axis_v": "1.0",
                "fog": "0",
                "unused": "a",
            },
            {
                "participant": "002",
                "axis_ap": "0.3",
                "axis_ml": "0.4",
                "axis_v": "0.9",
                "fog": "1",
                "unused": "b",
            },
        ],
    )
    schema = DataSchema(
        accelerometer_columns=("axis_ap", "axis_ml", "axis_v"),
        subject_id_column="participant",
        label_column="fog",
    )

    dataset = load_csv(csv_path, schema)

    assert dataset.schema == schema
    assert dataset.samples == ((0.1, -0.2, 1.0), (0.3, 0.4, 0.9))
    assert dataset.subject_ids == ("001", "002")
    assert dataset.labels == (0, 1)
    assert len(dataset) == 2


def test_load_csv_reports_all_missing_required_columns(tmp_path):
    csv_path = tmp_path / "missing_columns.csv"
    _write_csv(
        csv_path,
        ["subject", "acc_x"],
        [{"subject": "s1", "acc_x": "0.1"}],
    )
    schema = DataSchema(
        accelerometer_columns=("acc_x", "acc_y", "acc_z"),
        subject_id_column="subject",
        label_column="label",
    )

    with pytest.raises(DataValidationError, match="acc_y, acc_z, label"):
        load_csv(csv_path, schema)


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("fog", "must be an integer"),
        ("2", "Invalid label 2"),
        ("", "must be an integer"),
    ],
)
def test_load_csv_rejects_invalid_labels(tmp_path, label, message):
    csv_path = tmp_path / "invalid_label.csv"
    _write_csv(
        csv_path,
        ["subject", "acc_x", "acc_y", "acc_z", "label"],
        [
            {
                "subject": "s1",
                "acc_x": "0.1",
                "acc_y": "0.2",
                "acc_z": "0.3",
                "label": label,
            }
        ],
    )
    schema = DataSchema(
        accelerometer_columns=("acc_x", "acc_y", "acc_z"),
        subject_id_column="subject",
        label_column="label",
    )

    with pytest.raises(DataValidationError, match=message):
        load_csv(csv_path, schema)


def test_load_csv_accepts_configured_integer_labels(tmp_path):
    csv_path = tmp_path / "custom_labels.csv"
    _write_csv(
        csv_path,
        ["subject", "x", "y", "z", "state"],
        [{"subject": "s1", "x": "1", "y": "2", "z": "3", "state": "2"}],
    )
    schema = DataSchema(
        accelerometer_columns=("x", "y", "z"),
        subject_id_column="subject",
        label_column="state",
        allowed_labels=frozenset({1, 2}),
    )

    dataset = load_csv(csv_path, schema)

    assert dataset.labels == (2,)


@pytest.mark.parametrize(
    "schema_kwargs",
    [
        {"accelerometer_columns": ()},
        {"accelerometer_columns": ("acc_x", "acc_x")},
        {"accelerometer_columns": ("acc_x",), "subject_id_column": "acc_x"},
        {"allowed_labels": frozenset()},
    ],
)
def test_data_schema_rejects_invalid_configuration(schema_kwargs):
    defaults = {
        "accelerometer_columns": ("acc_x", "acc_y", "acc_z"),
        "subject_id_column": "subject",
        "label_column": "label",
    }

    with pytest.raises(DataValidationError):
        DataSchema(**(defaults | schema_kwargs))
