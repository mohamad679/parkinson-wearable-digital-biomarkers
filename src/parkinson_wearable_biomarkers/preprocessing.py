"""Deterministic, subject-safe windowing for accelerometer samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real

from parkinson_wearable_biomarkers.data import DataSchema, SensorDataset


class WindowingError(ValueError):
    """Raised when a windowing configuration or result is invalid."""


@dataclass(frozen=True, slots=True)
class WindowedDataset:
    """Fixed-size sensor windows with one subject ID and label per window."""

    schema: DataSchema
    windows: tuple[tuple[tuple[float, ...], ...], ...]
    subject_ids: tuple[str, ...]
    labels: tuple[int, ...]
    window_size: int

    def __post_init__(self) -> None:
        window_count = len(self.windows)
        if len(self.subject_ids) != window_count or len(self.labels) != window_count:
            raise WindowingError("Windows, subject IDs, and labels must have equal lengths")
        if self.window_size <= 0:
            raise WindowingError("Window size must be greater than zero")

        axis_count = len(self.schema.accelerometer_columns)
        if any(len(window) != self.window_size for window in self.windows):
            raise WindowingError(f"Every window must contain {self.window_size} samples")
        if any(len(sample) != axis_count for window in self.windows for sample in window):
            raise WindowingError(f"Every sample must contain {axis_count} accelerometer values")

    def __len__(self) -> int:
        """Return the number of complete windows."""
        return len(self.windows)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(window count, samples per window, accelerometer axes)``."""
        return (len(self), self.window_size, len(self.schema.accelerometer_columns))


def create_windows(
    dataset: SensorDataset,
    *,
    window_size: int,
    overlap_fraction: float = 0.0,
) -> WindowedDataset:
    """Create complete fixed-size windows without crossing subject boundaries.

    ``window_size`` is expressed in samples. ``overlap_fraction`` is converted to
    an integer number of samples by rounding down. Incomplete trailing samples in
    each contiguous subject run are discarded.

    A window's label is the most frequent sample label in that window. If labels
    are tied, the label that occurs first in the window is selected. This makes
    mixed-label handling deterministic without assigning meaning to label values.
    """
    normalized_window_size, step_size = _validate_window_parameters(
        window_size, overlap_fraction
    )
    windows: list[tuple[tuple[float, ...], ...]] = []
    subject_ids: list[str] = []
    labels: list[int] = []

    segment_start = 0
    while segment_start < len(dataset):
        subject_id = dataset.subject_ids[segment_start]
        segment_end = segment_start + 1
        while (
            segment_end < len(dataset) and dataset.subject_ids[segment_end] == subject_id
        ):
            segment_end += 1

        final_start = segment_end - normalized_window_size
        for start in range(segment_start, final_start + 1, step_size):
            end = start + normalized_window_size
            windows.append(dataset.samples[start:end])
            subject_ids.append(subject_id)
            labels.append(_majority_label(dataset.labels[start:end]))

        segment_start = segment_end

    return WindowedDataset(
        schema=dataset.schema,
        windows=tuple(windows),
        subject_ids=tuple(subject_ids),
        labels=tuple(labels),
        window_size=normalized_window_size,
    )


def _validate_window_parameters(window_size: int, overlap_fraction: float) -> tuple[int, int]:
    if isinstance(window_size, bool) or not isinstance(window_size, Integral):
        raise WindowingError("Window size must be an integer number of samples")

    normalized_window_size = int(window_size)
    if normalized_window_size <= 0:
        raise WindowingError("Window size must be greater than zero")

    if isinstance(overlap_fraction, bool) or not isinstance(overlap_fraction, Real):
        raise WindowingError("Overlap fraction must be a real number")

    normalized_overlap = float(overlap_fraction)
    if not 0.0 <= normalized_overlap < 1.0:
        raise WindowingError("Overlap fraction must be at least 0 and less than 1")

    overlap_samples = int(normalized_window_size * normalized_overlap)
    return normalized_window_size, normalized_window_size - overlap_samples


def _majority_label(labels: tuple[int, ...]) -> int:
    counts = Counter(labels)
    highest_count = max(counts.values())
    return next(label for label in labels if counts[label] == highest_count)
