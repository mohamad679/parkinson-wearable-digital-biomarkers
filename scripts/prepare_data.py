"""Create deterministic synthetic accelerometer CSV data for pipeline checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from numbers import Integral, Real
from pathlib import Path
from typing import Any


def generate_synthetic_csv(
    output_path: str | Path,
    *,
    subject_count: int = 6,
    samples_per_subject: int = 80,
    sampling_rate_hz: float = 20.0,
    event_segment_size: int = 20,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Write deterministic, non-clinical toy sensor data and return a summary."""
    subjects = _positive_integer(subject_count, "Subject count")
    samples = _positive_integer(samples_per_subject, "Samples per subject")
    segment_size = _positive_integer(event_segment_size, "Event segment size")
    seed = _non_negative_integer(random_seed, "Random seed")
    sampling_rate = _positive_real(sampling_rate_hz, "Sampling rate")
    if subjects < 2:
        raise ValueError("Subject count must be at least 2 for subject-aware validation")
    if samples < 2 * segment_size:
        raise ValueError(
            "Samples per subject must cover at least two event-label segments"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = random.Random(seed)
    fieldnames = ["subject_id", "acc_x", "acc_y", "acc_z", "label"]

    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for subject_index in range(subjects):
            subject_id = f"subject_{subject_index + 1:03d}"
            subject_phase = subject_index * 0.17
            subject_offset = subject_index * 0.01
            for sample_index in range(samples):
                label = (sample_index // segment_size) % 2
                frequency_hz = 1.0 + 2.0 * label
                angle = (
                    2.0 * math.pi * frequency_hz * sample_index / sampling_rate
                    + subject_phase
                )
                writer.writerow(
                    {
                        "subject_id": subject_id,
                        "acc_x": _format_sensor_value(
                            math.sin(angle)
                            + subject_offset
                            + generator.uniform(-0.02, 0.02)
                        ),
                        "acc_y": _format_sensor_value(
                            0.7 * math.cos(angle)
                            + generator.uniform(-0.02, 0.02)
                        ),
                        "acc_z": _format_sensor_value(
                            1.0
                            + 0.3 * math.sin(angle / 2.0)
                            + 0.15 * label
                            + generator.uniform(-0.02, 0.02)
                        ),
                        "label": label,
                    }
                )

    return {
        "mode": "synthetic",
        "output": str(output),
        "random_seed": seed,
        "row_count": subjects * samples,
        "sampling_rate_hz": sampling_rate,
        "subject_count": subjects,
    }


def _format_sensor_value(value: float) -> str:
    return f"{value:.8f}"


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _positive_real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return normalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic accelerometer data."
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        required=True,
        help="Explicitly select synthetic toy-data generation.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--samples-per-subject", type=int, default=80)
    parser.add_argument("--sampling-rate", type=float, default=20.0)
    parser.add_argument("--event-segment-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the synthetic data preparation command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = generate_synthetic_csv(
            arguments.output,
            subject_count=arguments.subjects,
            samples_per_subject=arguments.samples_per_subject,
            sampling_rate_hz=arguments.sampling_rate,
            event_segment_size=arguments.event_segment_size,
            random_seed=arguments.random_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
