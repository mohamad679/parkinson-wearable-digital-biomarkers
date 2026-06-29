"""Run deterministic subject-aware baselines and write benchmark JSON."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def run_baseline_pipeline(
    output_path: str | Path,
    *,
    input_path: str | Path | None = None,
    synthetic: bool = False,
    subject_count: int = 6,
    samples_per_subject: int = 80,
    event_segment_size: int = 20,
    sampling_rate_hz: float = 20.0,
    window_size: int = 20,
    overlap_fraction: float = 0.5,
    n_splits: int = 3,
    random_seed: int = 42,
    random_forest_estimators: int = 50,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    calibration_bins: int = 5,
    accelerometer_columns: tuple[str, ...] = ("acc_x", "acc_y", "acc_z"),
    subject_id_column: str = "subject_id",
    label_column: str = "label",
) -> dict[str, Any]:
    """Execute the full baseline pipeline and persist a benchmark summary."""
    if synthetic and input_path is not None:
        raise ValueError("Choose either synthetic data or an input CSV, not both")
    if not synthetic and input_path is None:
        raise ValueError("Provide an input CSV or select synthetic data")

    if synthetic:
        from scripts.prepare_data import generate_synthetic_csv

        with tempfile.TemporaryDirectory(prefix="parkinson-synthetic-") as directory:
            synthetic_path = Path(directory) / "synthetic_sensor_data.csv"
            generate_synthetic_csv(
                synthetic_path,
                subject_count=subject_count,
                samples_per_subject=samples_per_subject,
                sampling_rate_hz=sampling_rate_hz,
                event_segment_size=event_segment_size,
                random_seed=random_seed,
            )
            summary = _run_from_csv(
                synthetic_path,
                data_source="synthetic",
                sampling_rate_hz=sampling_rate_hz,
                window_size=window_size,
                overlap_fraction=overlap_fraction,
                n_splits=n_splits,
                random_seed=random_seed,
                random_forest_estimators=random_forest_estimators,
                thresholds=thresholds,
                calibration_bins=calibration_bins,
                accelerometer_columns=accelerometer_columns,
                subject_id_column=subject_id_column,
                label_column=label_column,
            )
    else:
        summary = _run_from_csv(
            Path(input_path),
            data_source=str(input_path),
            sampling_rate_hz=sampling_rate_hz,
            window_size=window_size,
            overlap_fraction=overlap_fraction,
            n_splits=n_splits,
            random_seed=random_seed,
            random_forest_estimators=random_forest_estimators,
            thresholds=thresholds,
            calibration_bins=calibration_bins,
            accelerometer_columns=accelerometer_columns,
            subject_id_column=subject_id_column,
            label_column=label_column,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_from_csv(
    input_path: Path,
    *,
    data_source: str,
    sampling_rate_hz: float,
    window_size: int,
    overlap_fraction: float,
    n_splits: int,
    random_seed: int,
    random_forest_estimators: int,
    thresholds: Sequence[float],
    calibration_bins: int,
    accelerometer_columns: tuple[str, ...],
    subject_id_column: str,
    label_column: str,
) -> dict[str, Any]:
    from parkinson_wearable_biomarkers.calibration import (
        brier_score,
        calibration_curve_data,
        expected_calibration_error,
    )
    from parkinson_wearable_biomarkers.data import DataSchema, load_csv
    from parkinson_wearable_biomarkers.evaluate import auprc, auroc, threshold_analysis
    from parkinson_wearable_biomarkers.features import extract_features
    from parkinson_wearable_biomarkers.models import fit_predict_fold
    from parkinson_wearable_biomarkers.preprocessing import create_windows
    from parkinson_wearable_biomarkers.validation import group_k_fold_splits

    schema = DataSchema(
        accelerometer_columns=accelerometer_columns,
        subject_id_column=subject_id_column,
        label_column=label_column,
    )
    sensor_data = load_csv(input_path, schema)
    windows = create_windows(
        sensor_data,
        window_size=window_size,
        overlap_fraction=overlap_fraction,
    )
    feature_data = extract_features(
        windows,
        sampling_rate_hz=sampling_rate_hz,
        rolling_window_size=min(3, window_size),
    )
    folds = group_k_fold_splits(feature_data, n_splits=n_splits)
    model_summaries: dict[str, Any] = {}

    for model_name in ("logistic_regression", "random_forest"):
        fold_predictions = tuple(
            fit_predict_fold(
                feature_data,
                fold,
                model_name=model_name,
                random_seed=random_seed,
                random_forest_estimators=random_forest_estimators,
            )
            for fold in folds
        )
        labels = tuple(label for prediction in fold_predictions for label in prediction.labels)
        probabilities = tuple(
            probability
            for prediction in fold_predictions
            for probability in prediction.probabilities
        )
        threshold_results = threshold_analysis(
            labels, probabilities, thresholds=thresholds
        )
        calibration_bins_data = calibration_curve_data(
            labels, probabilities, n_bins=calibration_bins
        )
        model_summaries[model_name] = {
            "auprc": auprc(labels, probabilities),
            "auroc": auroc(labels, probabilities),
            "brier_score": brier_score(labels, probabilities),
            "calibration_curve": [
                {
                    "bin_index": item.bin_index,
                    "count": item.count,
                    "lower_bound": item.lower_bound,
                    "mean_probability": item.mean_probability,
                    "observed_positive_rate": item.observed_positive_rate,
                    "upper_bound": item.upper_bound,
                }
                for item in calibration_bins_data
            ],
            "expected_calibration_error": expected_calibration_error(
                labels, probabilities, n_bins=calibration_bins
            ),
            "threshold_analysis": [
                _serialize_threshold_result(item) for item in threshold_results
            ],
        }

    return {
        "interpretation": (
            "Synthetic, non-diagnostic research benchmark; not evidence of clinical readiness."
        ),
        "models": model_summaries,
        "pipeline": {
            "data_source": data_source,
            "feature_count": len(feature_data.feature_names),
            "fold_count": len(folds),
            "overlap_fraction": overlap_fraction,
            "random_seed": random_seed,
            "sample_count": len(sensor_data),
            "sampling_rate_hz": sampling_rate_hz,
            "subject_count": len(set(sensor_data.subject_ids)),
            "window_count": len(windows),
            "window_size_samples": window_size,
        },
    }


def _serialize_threshold_result(result: Any) -> dict[str, Any]:
    matrix = result.confusion_matrix
    return {
        "confusion_matrix": {
            "false_negative": matrix.false_negative,
            "false_positive": matrix.false_positive,
            "true_negative": matrix.true_negative,
            "true_positive": matrix.true_positive,
        },
        "f1": result.f1,
        "precision": result.precision,
        "sensitivity": result.sensitivity,
        "specificity": result.specificity,
        "threshold": result.threshold,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic subject-aware baselines."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Input accelerometer CSV.")
    source.add_argument(
        "--synthetic", action="store_true", help="Use generated synthetic toy data."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/synthetic_benchmark.json")
    )
    parser.add_argument("--subjects", type=int, default=6)
    parser.add_argument("--samples-per-subject", type=int, default=80)
    parser.add_argument("--event-segment-size", type=int, default=20)
    parser.add_argument("--sampling-rate", type=float, default=20.0)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-forest-estimators", type=int, default=50)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--calibration-bins", type=int, default=5)
    parser.add_argument(
        "--accelerometer-columns", nargs="+", default=["acc_x", "acc_y", "acc_z"]
    )
    parser.add_argument("--subject-id-column", default="subject_id")
    parser.add_argument("--label-column", default="label")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the baseline benchmark command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    use_synthetic = arguments.synthetic or arguments.input is None
    try:
        summary = run_baseline_pipeline(
            arguments.output,
            input_path=arguments.input,
            synthetic=use_synthetic,
            subject_count=arguments.subjects,
            samples_per_subject=arguments.samples_per_subject,
            event_segment_size=arguments.event_segment_size,
            sampling_rate_hz=arguments.sampling_rate,
            window_size=arguments.window_size,
            overlap_fraction=arguments.overlap,
            n_splits=arguments.folds,
            random_seed=arguments.random_seed,
            random_forest_estimators=arguments.random_forest_estimators,
            thresholds=arguments.thresholds,
            calibration_bins=arguments.calibration_bins,
            accelerometer_columns=tuple(arguments.accelerometer_columns),
            subject_id_column=arguments.subject_id_column,
            label_column=arguments.label_column,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
