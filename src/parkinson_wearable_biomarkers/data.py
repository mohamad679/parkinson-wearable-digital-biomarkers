"""Schema validation and CSV loading for wearable accelerometer data."""

from __future__ import annotations

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path


class DataValidationError(ValueError):
    """Raised when wearable sensor data does not satisfy the configured schema."""


@dataclass(frozen=True, slots=True)
class DataSchema:
    """Column names and permitted labels for an accelerometer dataset."""

    accelerometer_columns: tuple[str, ...]
    subject_id_column: str
    label_column: str
    allowed_labels: frozenset[int] = frozenset({0, 1})

    def __post_init__(self) -> None:
        if isinstance(self.accelerometer_columns, str):
            raise DataValidationError("accelerometer_columns must be a sequence of column names")

        accelerometer_columns = tuple(self.accelerometer_columns)
        allowed_labels = frozenset(self.allowed_labels)
        all_columns = (*accelerometer_columns, self.subject_id_column, self.label_column)

        if not accelerometer_columns:
            raise DataValidationError("At least one accelerometer column is required")
        if any(not isinstance(column, str) or not column.strip() for column in all_columns):
            raise DataValidationError("Configured column names must be non-empty strings")

        duplicate_columns = sorted(
            column for column in set(all_columns) if all_columns.count(column) > 1
        )
        if duplicate_columns:
            duplicates = ", ".join(duplicate_columns)
            raise DataValidationError(f"Configured columns must be distinct: {duplicates}")

        if not allowed_labels:
            raise DataValidationError("At least one allowed label is required")
        if any(not isinstance(label, int) or isinstance(label, bool) for label in allowed_labels):
            raise DataValidationError("Allowed labels must be integers")

        object.__setattr__(self, "accelerometer_columns", accelerometer_columns)
        object.__setattr__(self, "allowed_labels", allowed_labels)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return every CSV column required by this schema."""
        return (*self.accelerometer_columns, self.subject_id_column, self.label_column)


@dataclass(frozen=True, slots=True)
class SensorDataset:
    """Validated accelerometer samples and their subject-level metadata."""

    schema: DataSchema
    samples: tuple[tuple[float, ...], ...]
    subject_ids: tuple[str, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        row_count = len(self.samples)
        if len(self.subject_ids) != row_count or len(self.labels) != row_count:
            raise DataValidationError("Samples, subject IDs, and labels must have equal lengths")

        expected_width = len(self.schema.accelerometer_columns)
        if any(len(sample) != expected_width for sample in self.samples):
            raise DataValidationError(
                f"Each sample must contain {expected_width} accelerometer values"
            )

    def __len__(self) -> int:
        """Return the number of sensor samples."""
        return len(self.samples)


def load_csv(path: str | Path, schema: DataSchema) -> SensorDataset:
    """Load and validate a CSV file according to ``schema``.

    Subject identifiers are preserved as strings, labels are parsed as integers,
    and accelerometer values are parsed as floats.
    """
    csv_path = Path(path)
    samples: list[tuple[float, ...]] = []
    subject_ids: list[str] = []
    labels: list[int] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        _validate_header(reader.fieldnames, schema)

        for line_number, row in enumerate(reader, start=2):
            subject_ids.append(
                _read_subject_id(row.get(schema.subject_id_column), line_number)
            )
            labels.append(
                _read_label(row.get(schema.label_column), schema.allowed_labels, line_number)
            )
            samples.append(
                tuple(
                    _read_acceleration(row.get(column), column, line_number)
                    for column in schema.accelerometer_columns
                )
            )

    if not samples:
        raise DataValidationError("CSV file contains no data rows")

    return SensorDataset(
        schema=schema,
        samples=tuple(samples),
        subject_ids=tuple(subject_ids),
        labels=tuple(labels),
    )


def _validate_header(fieldnames: list[str] | None, schema: DataSchema) -> None:
    if fieldnames is None:
        raise DataValidationError("CSV file is missing a header row")

    missing_columns = sorted(set(schema.required_columns).difference(fieldnames))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataValidationError(f"CSV file is missing required columns: {missing}")


def _read_subject_id(value: str | None, line_number: int) -> str:
    subject_id = value.strip() if value is not None else ""
    if not subject_id:
        raise DataValidationError(f"Missing subject ID on CSV line {line_number}")
    return subject_id


def _read_label(value: str | None, allowed_labels: Collection[int], line_number: int) -> int:
    label_text = value.strip() if value is not None else ""
    try:
        label = int(label_text)
    except ValueError as error:
        raise DataValidationError(
            f"Label on CSV line {line_number} must be an integer: {label_text!r}"
        ) from error

    if label not in allowed_labels:
        expected = ", ".join(str(item) for item in sorted(allowed_labels))
        raise DataValidationError(
            f"Invalid label {label} on CSV line {line_number}; expected one of: {expected}"
        )
    return label


def _read_acceleration(value: str | None, column: str, line_number: int) -> float:
    acceleration_text = value.strip() if value is not None else ""
    try:
        return float(acceleration_text)
    except ValueError as error:
        raise DataValidationError(
            f"Accelerometer value in column {column!r} on CSV line {line_number} "
            f"must be numeric: {acceleration_text!r}"
        ) from error
