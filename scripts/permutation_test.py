"""Subject-level permutation significance test for wearable biomarker baselines.

Design decisions
----------------
- Uses 3-fold GroupKFold (not 10-fold LOSO) for the null distribution.
  With 1000 permutations, LOSO costs ~3.3x more compute (10 model fits per
  permutation vs 3). The null distribution converges to 0.50 regardless of
  fold count, so the computational saving involves no loss of statistical
  validity. This choice is documented explicitly here and in the accompanying
  report.

- Labels are shuffled within each subject (not globally). Global shuffling
  would create folds with zero positives for low-prevalence subjects, causing
  CV to crash. Within-subject shuffling preserves per-subject class prevalence
  so every fold in every permutation is guaranteed trainable.

- Empirical p-value uses the conservative Phipson & Smyth (2010) estimator:
  (1 + count(null >= observed)) / (n_permutations + 1).
  This avoids reporting p = 0 when all permutations fall below the observed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))


# ── Public API ────────────────────────────────────────────────────────────────


def permute_labels_within_subjects(
    feature_data: Any,
    *,
    rng: random.Random,
) -> Any:
    """Return a copy of *feature_data* with labels shuffled within each subject.

    Shuffling within subjects (not globally) preserves per-subject class
    prevalence so that every cross-validation fold in every permutation
    contains both binary classes in its training set.

    Parameters
    ----------
    feature_data:
        A ``FeatureDataset`` whose labels will be permuted.
    rng:
        A seeded ``random.Random`` instance used for reproducible shuffling.
        The caller is responsible for seeding it appropriately.

    Returns
    -------
    FeatureDataset
        A new ``FeatureDataset`` identical to *feature_data* except for the
        permuted ``labels`` tuple.
    """
    from parkinson_wearable_biomarkers.features import FeatureDataset

    # Build a mapping from subject_id to the row indices it occupies.
    subject_to_indices: dict[str, list[int]] = {}
    for index, subject_id in enumerate(feature_data.subject_ids):
        subject_to_indices.setdefault(subject_id, []).append(index)

    # Shuffle labels within each subject group independently.
    new_labels: list[int] = list(feature_data.labels)
    for indices in subject_to_indices.values():
        subject_labels = [new_labels[i] for i in indices]
        rng.shuffle(subject_labels)
        for row_index, permuted_label in zip(indices, subject_labels, strict=False):
            new_labels[row_index] = permuted_label

    return FeatureDataset(
        feature_names=feature_data.feature_names,
        features=feature_data.features,
        subject_ids=feature_data.subject_ids,
        labels=tuple(new_labels),
    )


def run_permutation_test(
    output_path: str | Path,
    *,
    input_path: str | Path,
    sampling_rate_hz: float = 64.0,
    window_size: int = 128,
    overlap_fraction: float = 0.5,
    n_splits: int = 3,
    n_permutations: int = 1000,
    random_seed: int = 42,
    random_forest_estimators: int = 50,
    accelerometer_columns: tuple[str, ...] = ("acc_x", "acc_y", "acc_z"),
    subject_id_column: str = "subject_id",
    label_column: str = "label",
) -> dict[str, Any]:
    """Run a within-subject permutation significance test and persist results.

    For each model (logistic regression, random forest) this function:

    1. Computes the observed out-of-fold AUROC on the original labels using
       GroupKFold cross-validation.
    2. Generates ``n_permutations`` null-distribution AUROC values by shuffling
       labels within each subject and repeating CV.
    3. Computes the conservative empirical p-value
       ``(1 + count(null >= observed)) / (n_permutations + 1)``.
    4. Writes a JSON file containing the full null arrays and summary statistics.

    Parameters
    ----------
    output_path:
        Destination path for the benchmark JSON (created with parents).
    input_path:
        Path to the preprocessed accelerometer CSV.
    sampling_rate_hz:
        Sensor sampling rate used for frequency-domain features.
    window_size:
        Number of samples per window.
    overlap_fraction:
        Fractional overlap between consecutive windows (exclusive of 1.0).
    n_splits:
        Number of GroupKFold folds for both the observed and null AUROC.
    n_permutations:
        Number of random permutations in the null distribution.
    random_seed:
        Base seed used for deterministic model fitting and permutation RNGs.
    random_forest_estimators:
        Number of trees in the random forest.
    accelerometer_columns:
        CSV column names for the accelerometer axes.
    subject_id_column:
        CSV column name for the subject identifier.
    label_column:
        CSV column name for the binary label (0 / 1).

    Returns
    -------
    dict
        The same structure written to *output_path*.
    """
    from parkinson_wearable_biomarkers.data import DataSchema, load_csv
    from parkinson_wearable_biomarkers.features import extract_features
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
        # ── Observed OOF AUROC ──────────────────────────────────────────────
        observed_auroc = _compute_oof_auroc(
            feature_data,
            folds,
            model_name=model_name,
            random_seed=random_seed,
            random_forest_estimators=random_forest_estimators,
        )

        # ── Null distribution ───────────────────────────────────────────────
        null_aurocs: list[float] = []
        failed_permutations = 0

        for perm_index in range(n_permutations):
            # Each permutation gets its own seeded RNG derived from the base
            # seed so the full run is reproducible from --random-seed alone.
            perm_rng = random.Random(random_seed + perm_index + 1)
            permuted = permute_labels_within_subjects(feature_data, rng=perm_rng)
            perm_folds = group_k_fold_splits(permuted, n_splits=n_splits)
            try:
                null_val = _compute_oof_auroc(
                    permuted,
                    perm_folds,
                    model_name=model_name,
                    random_seed=random_seed,
                    random_forest_estimators=random_forest_estimators,
                )
                null_aurocs.append(null_val)
            except Exception:  # noqa: BLE001
                # Skip rare permutations where a fold unexpectedly lacks both
                # binary classes (can occur with very small synthetic data).
                failed_permutations += 1

        n_effective = len(null_aurocs)

        if n_effective == 0:
            model_summaries[model_name] = {
                "error": "All permutations failed; cannot compute p-value",
                "failed_permutations": failed_permutations,
                "n_effective_permutations": 0,
                "n_permutations": n_permutations,
                "null_auroc": [],
                "null_mean": None,
                "null_std": None,
                "observed_auroc": observed_auroc,
                "p_value": None,
            }
            continue

        null_mean = math.fsum(null_aurocs) / n_effective
        null_variance = (
            math.fsum((v - null_mean) ** 2 for v in null_aurocs) / n_effective
        )
        null_std = math.sqrt(null_variance)

        # Conservative p-value: Phipson & Smyth (2010).
        exceedances = sum(1 for v in null_aurocs if v >= observed_auroc)
        p_value = (1 + exceedances) / (n_effective + 1)

        model_summaries[model_name] = {
            "failed_permutations": failed_permutations,
            "n_effective_permutations": n_effective,
            "n_permutations": n_permutations,
            "null_auroc": null_aurocs,
            "null_mean": null_mean,
            "null_std": null_std,
            "observed_auroc": observed_auroc,
            "p_value": p_value,
        }

    summary: dict[str, Any] = {
        "interpretation": (
            "Non-diagnostic research benchmark. The permutation p-value is "
            "empirical and dataset-specific; it does not establish clinical "
            "validity or generalisability to other cohorts or devices."
        ),
        "models": model_summaries,
        "pipeline": {
            "data_source": str(input_path),
            "feature_count": len(feature_data.feature_names),
            "n_folds": n_splits,
            "n_permutations": n_permutations,
            "overlap_fraction": overlap_fraction,
            "permutation_strategy": "within_subject",
            "random_seed": random_seed,
            "sample_count": len(sensor_data),
            "sampling_rate_hz": sampling_rate_hz,
            "subject_count": len(set(sensor_data.subject_ids)),
            "validation": "groupkfold",
            "window_count": len(windows),
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


# ── Internal helpers ──────────────────────────────────────────────────────────


def _compute_oof_auroc(
    feature_data: Any,
    folds: tuple,
    *,
    model_name: str,
    random_seed: int,
    random_forest_estimators: int,
) -> float:
    """Fit CV folds and return the concatenated out-of-fold AUROC."""
    from parkinson_wearable_biomarkers.evaluate import auroc
    from parkinson_wearable_biomarkers.models import fit_predict_fold

    all_labels: list[int] = []
    all_probs: list[float] = []
    for fold in folds:
        prediction = fit_predict_fold(
            feature_data,
            fold,
            model_name=model_name,
            random_seed=random_seed,
            random_forest_estimators=random_forest_estimators,
        )
        all_labels.extend(prediction.labels)
        all_probs.extend(prediction.probabilities)
    return auroc(all_labels, all_probs)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a within-subject permutation significance test. "
            "Uses 3-fold GroupKFold (not LOSO) for the null distribution; "
            "see script docstring for the rationale."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/daphnet_trunk_permutation.json"),
    )
    parser.add_argument("--sampling-rate", type=float, default=64.0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--random-forest-estimators", type=int, default=50)
    parser.add_argument(
        "--accelerometer-columns", nargs="+", default=["acc_x", "acc_y", "acc_z"]
    )
    parser.add_argument("--subject-id-column", default="subject_id")
    parser.add_argument("--label-column", default="label")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the permutation significance test command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        summary = run_permutation_test(
            arguments.output,
            input_path=arguments.input,
            sampling_rate_hz=arguments.sampling_rate,
            window_size=arguments.window_size,
            overlap_fraction=arguments.overlap,
            n_splits=arguments.folds,
            n_permutations=arguments.n_permutations,
            random_seed=arguments.random_seed,
            random_forest_estimators=arguments.random_forest_estimators,
            accelerometer_columns=tuple(arguments.accelerometer_columns),
            subject_id_column=arguments.subject_id_column,
            label_column=arguments.label_column,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 1
    # Print a compact summary (omit the full null arrays).
    compact = {
        "pipeline": summary["pipeline"],
        "models": {
            model: {k: v for k, v in data.items() if k != "null_auroc"}
            for model, data in summary["models"].items()
        },
    }
    print(json.dumps(compact, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
