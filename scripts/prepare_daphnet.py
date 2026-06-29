"""Convert a local Daphnet directory or ZIP into the project CSV schema."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

_SENSOR_COLUMNS = {
    "ankle": (1, 2, 3),
    "thigh": (4, 5, 6),
    "trunk": (7, 8, 9),
}
_RECORDING_NAME = re.compile(r"^(S\d+)R\d+\.txt$", re.IGNORECASE)


class DaphnetConversionError(ValueError):
    """Raised when local Daphnet input cannot be converted safely."""


def convert_daphnet(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sensor: str = "trunk",
) -> dict[str, Any]:
    """Convert Daphnet recordings from a directory or ZIP without network access."""
    normalized_sensor = sensor.lower()
    if normalized_sensor not in _SENSOR_COLUMNS:
        choices = ", ".join(sorted(_SENSOR_COLUMNS))
        raise DaphnetConversionError(f"Sensor must be one of: {choices}")

    source = Path(input_path)
    if not source.exists():
        raise DaphnetConversionError(f"Daphnet input does not exist: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["subject_id", "acc_x", "acc_y", "acc_z", "label"],
            )
            writer.writeheader()
            if source.is_dir():
                counts = _convert_directory(source, writer, normalized_sensor)
            elif source.is_file() and zipfile.is_zipfile(source):
                counts = _convert_zip(source, writer, normalized_sensor)
            else:
                raise DaphnetConversionError(
                    "Daphnet input must be a directory or readable ZIP archive"
                )

        if counts["recording_count"] == 0:
            raise DaphnetConversionError(
                "No Daphnet recording files matching S<subject>R<recording>.txt were found"
            )
        if counts["rows_written"] == 0:
            raise DaphnetConversionError("No annotation 1 or 2 rows were found")
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return {
        "negative_no_freeze": counts["negative_no_freeze"],
        "output": str(output),
        "positive_freeze": counts["positive_freeze"],
        "rows_written": counts["rows_written"],
        "sensor": normalized_sensor,
        "subject_count": len(counts["subject_ids"]),
    }


def _convert_directory(
    directory: Path,
    writer: csv.DictWriter,
    sensor: str,
) -> dict[str, Any]:
    recordings = []
    for path in sorted(directory.rglob("*.txt")):
        subject_id = _subject_id(path.name)
        if subject_id is not None:
            recordings.append((path, subject_id))

    counts = _empty_counts()
    for path, subject_id in recordings:
        with path.open("r", encoding="utf-8") as stream:
            _convert_recording(
                stream,
                source_name=str(path),
                subject_id=subject_id,
                sensor=sensor,
                writer=writer,
                counts=counts,
            )
    counts["recording_count"] = len(recordings)
    return counts


def _convert_zip(
    archive_path: Path,
    writer: csv.DictWriter,
    sensor: str,
) -> dict[str, Any]:
    counts = _empty_counts()
    with zipfile.ZipFile(archive_path) as archive:
        recordings = []
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            subject_id = _subject_id(Path(member.filename).name)
            if subject_id is not None:
                recordings.append((member, subject_id))

        for member, subject_id in recordings:
            with archive.open(member) as binary_stream:
                with io.TextIOWrapper(binary_stream, encoding="utf-8") as stream:
                    _convert_recording(
                        stream,
                        source_name=f"{archive_path}:{member.filename}",
                        subject_id=subject_id,
                        sensor=sensor,
                        writer=writer,
                        counts=counts,
                    )
        counts["recording_count"] = len(recordings)
    return counts


def _convert_recording(
    stream: TextIO | Iterable[str],
    *,
    source_name: str,
    subject_id: str,
    sensor: str,
    writer: csv.DictWriter,
    counts: dict[str, Any],
) -> None:
    sensor_columns = _SENSOR_COLUMNS[sensor]
    subject_has_rows = False
    for line_number, line in enumerate(stream, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 11:
            raise DaphnetConversionError(
                f"{source_name}:{line_number}: expected 11 columns, found {len(fields)}"
            )

        try:
            measurements = tuple(float(value) for value in fields[:10])
            annotation_value = float(fields[10])
        except ValueError as error:
            raise DaphnetConversionError(
                f"{source_name}:{line_number}: all columns must be numeric"
            ) from error
        if any(not math.isfinite(value) for value in (*measurements, annotation_value)):
            raise DaphnetConversionError(
                f"{source_name}:{line_number}: columns must contain finite values"
            )
        if not annotation_value.is_integer():
            raise DaphnetConversionError(
                f"{source_name}:{line_number}: annotation must be 0, 1, or 2"
            )

        annotation = int(annotation_value)
        if annotation == 0:
            continue
        if annotation not in {1, 2}:
            raise DaphnetConversionError(
                f"{source_name}:{line_number}: annotation must be 0, 1, or 2"
            )

        label = annotation - 1
        writer.writerow(
            {
                "subject_id": subject_id,
                "acc_x": _format_number(measurements[sensor_columns[0]]),
                "acc_y": _format_number(measurements[sensor_columns[1]]),
                "acc_z": _format_number(measurements[sensor_columns[2]]),
                "label": label,
            }
        )
        counts["rows_written"] += 1
        if label == 0:
            counts["negative_no_freeze"] += 1
        else:
            counts["positive_freeze"] += 1
        subject_has_rows = True

    if subject_has_rows:
        counts["subject_ids"].add(subject_id)


def _subject_id(file_name: str) -> str | None:
    match = _RECORDING_NAME.fullmatch(file_name)
    return match.group(1).upper() if match else None


def _format_number(value: float) -> str:
    return format(value, ".12g")


def _empty_counts() -> dict[str, Any]:
    return {
        "negative_no_freeze": 0,
        "positive_freeze": 0,
        "recording_count": 0,
        "rows_written": 0,
        "subject_ids": set(),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local Daphnet directory or ZIP to project CSV format."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sensor", choices=sorted(_SENSOR_COLUMNS), default="trunk"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the local Daphnet conversion command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = convert_daphnet(
            arguments.input,
            arguments.output,
            sensor=arguments.sensor,
        )
    except (OSError, DaphnetConversionError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
