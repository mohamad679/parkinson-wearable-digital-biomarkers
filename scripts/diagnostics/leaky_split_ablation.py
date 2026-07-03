# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# FOR DIAGNOSTIC / DEMONSTRATION PURPOSES ONLY — NOT A VALID EVALUATION METHOD
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""Leaky vs. honest split ablation — diagnostic script.

PURPOSE
-------
This script demonstrates the magnitude of the subject-leakage bias described
in the project README and in docs/daphnet_leakage_ablation.md.

It intentionally uses StratifiedKFold, which ignores subject identity and
allows windows from the same recording session (same person, same sensor
placement, same gait signature) to appear in both the training and test sets.
That is *the* classical methodological error that subject-aware validation
prevents.

The inflated metrics produced here are NOT this project's benchmark result.
They exist only to attach a measured number to the leakage warning so that
readers can see exactly how much optimism the error introduces.

DO NOT USE THIS SCRIPT TO EVALUATE MODEL PERFORMANCE.
DO NOT CITE ITS NUMBERS AS A BENCHMARK.
DO NOT ADD A --leaky FLAG TO run_baselines.py — THAT RISKS ACCIDENTAL MISUSE.

REFERENCE
---------
docs/daphnet_leakage_ablation.md contains the full comparison table and
methodological explanation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))


# ── Public API ────────────────────────────────────────────────────────────────


def run_leaky_ablation(
    output_path: str | Path,
    *,
    input_path: str | Path,
    sampling_rate_hz: float = 64.0,
    window_size: int = 128,
    overlap_fraction: float = 0.5,
    n_splits: int = 3,
    random_seed: int = 42,
    random_forest_estimators: int = 50,
    accelerometer_columns: tuple[str, ...] = ("acc_x", "acc_y", "acc_z"),
    subject_id_column: str = "subject_id",
    label_column: str = "label",
) -> dict[str, Any]:
    """Run the leaky-split ablation and persist diagnostic results.

    THIS IS A DIAGNOSTIC FUNCTION. IT INTENTIONALLY USES AN INVALID EVALUATION
    STRATEGY (StratifiedKFold, ignoring subject identity) TO DEMONSTRATE
    SUBJECT-LEAKAGE BIAS. DO NOT USE THESE RESULTS AS A PERFORMANCE ESTIMATE.

    Parameters
    ----------
    output_path:
        Destination path for the diagnostic JSON.
    input_path:
        Path to the preprocessed accelerometer CSV.
    sampling_rate_hz:
        Sensor sampling rate (must match the honest benchmark).
    window_size:
        Samples per window (must match the honest benchmark).
    overlap_fraction:
        Window overlap fraction (must match the honest benchmark).
    n_splits:
        Number of StratifiedKFold folds (must match the honest benchmark).
    random_seed:
        Random seed for model fitting and StratifiedKFold shuffle.
    random_forest_estimators:
        Number of trees in the random forest (must match the honest benchmark).
    accelerometer_columns, subject_id_column, label_column:
        Column names (must match the honest benchmark).
    """
    # Delay imports so this module can be imported without sklearn on path.
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    from parkinson_wearable_biomarkers.calibration import (  # noqa: PLC0415
        brier_score,
        expected_calibration_error,
    )
    from parkinson_wearable_biomarkers.data import DataSchema, load_csv  # noqa: PLC0415
    from parkinson_wearable_biomarkers.evaluate import auprc, auroc  # noqa: PLC0415
    from parkinson_wearable_biomarkers.features import extract_features  # noqa: PLC0415
    from parkinson_wearable_biomarkers.models import (  # noqa: PLC0415
        fit_logistic_regression,
        fit_random_forest,
    )
    from parkinson_wearable_biomarkers.preprocessing import create_windows  # noqa: PLC0415

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

    n_windows = len(feature_data)
    labels_list = list(feature_data.labels)

    # StratifiedKFold: ignores subject_ids entirely.
    # This is intentionally wrong — it is the error being demonstrated.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    model_summaries: dict[str, Any] = {}

    fitters = {
        "logistic_regression": lambda train_idx: fit_logistic_regression(
            feature_data,
            train_indices=train_idx,
            random_seed=random_seed,
        ),
        "random_forest": lambda train_idx: fit_random_forest(
            feature_data,
            train_indices=train_idx,
            random_seed=random_seed,
            n_estimators=random_forest_estimators,
        ),
    }

    for model_name, fitter in fitters.items():
        all_labels: list[int] = []
        all_probs: list[float] = []

        for train_indices, test_indices in skf.split(range(n_windows), labels_list):
            fitted_model = fitter(list(train_indices))
            fold_probs = fitted_model.predict_positive_probabilities(
                feature_data, indices=list(test_indices)
            )
            fold_labels = [labels_list[i] for i in test_indices]
            all_labels.extend(fold_labels)
            all_probs.extend(fold_probs)

        oof_auroc = auroc(all_labels, all_probs)
        oof_auprc = auprc(all_labels, all_probs)
        oof_brier = brier_score(all_labels, all_probs)
        oof_ece = expected_calibration_error(all_labels, all_probs, n_bins=5)

        model_summaries[model_name] = {
            "auroc": oof_auroc,
            "auprc": oof_auprc,
            "brier_score": oof_brier,
            "expected_calibration_error": oof_ece,
        }

    summary: dict[str, Any] = {
        "DIAGNOSTIC_WARNING": (
            "THIS JSON WAS PRODUCED BY AN INTENTIONALLY INVALID EVALUATION. "
            "StratifiedKFold (subject-unaware) was used, which allows windows "
            "from the same subject to appear in both training and test sets. "
            "These numbers overestimate generalization and MUST NOT be cited "
            "as benchmark results."
        ),
        "models": model_summaries,
        "pipeline": {
            "data_source": str(input_path),
            "feature_count": len(feature_data.feature_names),
            "n_folds": n_splits,
            "overlap_fraction": overlap_fraction,
            "random_seed": random_seed,
            "sample_count": len(sensor_data),
            "sampling_rate_hz": sampling_rate_hz,
            "split_strategy": "StratifiedKFold_LEAKY_SUBJECT_UNAWARE",
            "subject_count": len(set(sensor_data.subject_ids)),
            "window_count": n_windows,
            "window_size_samples": window_size,
        },
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DIAGNOSTIC ONLY — NOT A VALID EVALUATION METHOD.\n"
            "Demonstrates subject-leakage bias by running StratifiedKFold "
            "(ignoring subject identity) on the same feature matrix used by "
            "the honest subject-aware benchmark. Compare output numbers to "
            "docs/daphnet_leakage_ablation.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/daphnet_trunk_leaky_benchmark.json"),
    )
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-forest-estimators", type=int, default=50)
    parser.add_argument("--accelerometer-columns", nargs="+", default=["acc_x", "acc_y", "acc_z"])
    parser.add_argument("--subject-id-column", default="subject_id")
    parser.add_argument("--label-column", default="label")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the leaky-split diagnostic command."""
    print(
        "\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        "FOR DIAGNOSTIC / DEMONSTRATION PURPOSES ONLY\n"
        "NOT A VALID EVALUATION METHOD — DO NOT CITE THESE NUMBERS\n"
        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
        file=sys.stderr,
    )
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = run_leaky_ablation(
            arguments.output,
            input_path=arguments.input,
            sampling_rate_hz=arguments.sampling_rate,
            window_size=arguments.window_size,
            overlap_fraction=arguments.overlap,
            n_splits=arguments.folds,
            random_seed=arguments.random_seed,
            random_forest_estimators=arguments.random_forest_estimators,
            accelerometer_columns=tuple(arguments.accelerometer_columns),
            subject_id_column=arguments.subject_id_column,
            label_column=arguments.label_column,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 1

    compact = {k: v for k, v in summary.items() if k != "DIAGNOSTIC_WARNING"}
    compact["DIAGNOSTIC_WARNING"] = summary["DIAGNOSTIC_WARNING"]
    print(json.dumps(compact, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
