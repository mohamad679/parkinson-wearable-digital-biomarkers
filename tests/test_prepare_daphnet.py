import csv
import json
import zipfile

import pytest

from scripts.prepare_daphnet import convert_daphnet, main


def _row(
    time_ms,
    *,
    ankle=(1, 2, 3),
    thigh=(4, 5, 6),
    trunk=(7, 8, 9),
    annotation=1,
):
    values = (time_ms, *ankle, *thigh, *trunk, annotation)
    return " ".join(str(value) for value in values)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def test_directory_conversion_maps_annotations_subjects_and_trunk_columns(tmp_path):
    input_directory = tmp_path / "daphnet"
    input_directory.mkdir()
    (input_directory / "README.txt").write_text("not a recording", encoding="utf-8")
    (input_directory / "S01R01.txt").write_text(
        "\n".join(
            [
                _row(0, trunk=(70, 80, 90), annotation=0),
                _row(1, trunk=(71, 81, 91), annotation=1),
                _row(2, trunk=(72, 82, 92), annotation=2),
            ]
        ),
        encoding="utf-8",
    )
    (input_directory / "S02R03.txt").write_text(
        _row(3, trunk=(73, 83, 93), annotation=1),
        encoding="utf-8",
    )
    output = tmp_path / "converted.csv"

    summary = convert_daphnet(input_directory, output)

    assert _read_csv(output) == [
        {
            "subject_id": "S01",
            "acc_x": "71",
            "acc_y": "81",
            "acc_z": "91",
            "label": "0",
        },
        {
            "subject_id": "S01",
            "acc_x": "72",
            "acc_y": "82",
            "acc_z": "92",
            "label": "1",
        },
        {
            "subject_id": "S02",
            "acc_x": "73",
            "acc_y": "83",
            "acc_z": "93",
            "label": "0",
        },
    ]
    assert summary == {
        "negative_no_freeze": 2,
        "output": str(output),
        "positive_freeze": 1,
        "rows_written": 3,
        "sensor": "trunk",
        "subject_count": 2,
    }


@pytest.mark.parametrize(
    ("sensor", "expected"),
    [
        ("ankle", ("11", "12", "13")),
        ("thigh", ("21", "22", "23")),
        ("trunk", ("31", "32", "33")),
    ],
)
def test_sensor_selection_uses_documented_column_triplets(tmp_path, sensor, expected):
    input_directory = tmp_path / sensor
    input_directory.mkdir()
    (input_directory / "S03R01.txt").write_text(
        _row(
            0,
            ankle=(11, 12, 13),
            thigh=(21, 22, 23),
            trunk=(31, 32, 33),
            annotation=2,
        ),
        encoding="utf-8",
    )
    output = tmp_path / f"{sensor}.csv"

    summary = convert_daphnet(input_directory, output, sensor=sensor)

    converted = _read_csv(output)
    assert (converted[0]["acc_x"], converted[0]["acc_y"], converted[0]["acc_z"]) == expected
    assert converted[0]["label"] == "1"
    assert summary["sensor"] == sensor


def test_zip_conversion_reads_nested_recordings_without_extracting(tmp_path):
    archive_path = tmp_path / "daphnet.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/README.txt", "citation text")
        archive.writestr(
            "dataset/S04R01.txt",
            "\n".join([_row(0, annotation=0), _row(1, ankle=(14, 15, 16), annotation=1)]),
        )
        archive.writestr(
            "dataset/nested/S05R02.txt",
            _row(2, ankle=(24, 25, 26), annotation=2),
        )
    output = tmp_path / "from_zip.csv"

    summary = convert_daphnet(archive_path, output, sensor="ankle")

    assert _read_csv(output) == [
        {
            "subject_id": "S04",
            "acc_x": "14",
            "acc_y": "15",
            "acc_z": "16",
            "label": "0",
        },
        {
            "subject_id": "S05",
            "acc_x": "24",
            "acc_y": "25",
            "acc_z": "26",
            "label": "1",
        },
    ]
    assert summary["rows_written"] == 2
    assert summary["subject_count"] == 2


def test_command_prints_required_json_summary(tmp_path, capsys):
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "S06R01.txt").write_text(
        "\n".join([_row(0, annotation=1), _row(1, annotation=2)]),
        encoding="utf-8",
    )
    output = tmp_path / "output.csv"

    exit_code = main(
        ["--input", str(input_directory), "--output", str(output), "--sensor", "thigh"]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["rows_written"] == 2
    assert summary["negative_no_freeze"] == 1
    assert summary["positive_freeze"] == 1
    assert summary["sensor"] == "thigh"
    assert summary["output"] == str(output)
