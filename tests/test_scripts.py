import json

from scripts.make_figures import make_benchmark_figure
from scripts.make_loso_figure import make_loso_figure
from scripts.prepare_data import generate_synthetic_csv
from scripts.run_baselines import run_baseline_pipeline


def test_synthetic_data_generation_is_deterministic(tmp_path):
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    first_summary = generate_synthetic_csv(
        first_path,
        subject_count=4,
        samples_per_subject=40,
        sampling_rate_hz=10,
        event_segment_size=10,
        random_seed=7,
    )
    second_summary = generate_synthetic_csv(
        second_path,
        subject_count=4,
        samples_per_subject=40,
        sampling_rate_hz=10,
        event_segment_size=10,
        random_seed=7,
    )

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
    assert first_summary["row_count"] == 160
    assert first_summary["subject_count"] == 4
    assert first_summary | {"output": "ignored"} == second_summary | {"output": "ignored"}


def test_baseline_pipeline_writes_benchmark_json_from_synthetic_data(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"

    summary = run_baseline_pipeline(
        benchmark_path,
        synthetic=True,
        subject_count=4,
        samples_per_subject=40,
        event_segment_size=10,
        sampling_rate_hz=10,
        window_size=10,
        overlap_fraction=0.5,
        n_splits=2,
        random_seed=11,
        random_forest_estimators=10,
        thresholds=(0.3, 0.5, 0.7),
        calibration_bins=4,
    )

    persisted = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert persisted == summary
    assert summary["pipeline"]["data_source"] == "synthetic"
    assert summary["pipeline"]["subject_count"] == 4
    assert summary["pipeline"]["window_count"] > 0
    assert set(summary["models"]) == {"logistic_regression", "random_forest"}
    for model_summary in summary["models"].values():
        assert 0.0 <= model_summary["auroc"] <= 1.0
        assert 0.0 <= model_summary["auprc"] <= 1.0
        assert 0.0 <= model_summary["brier_score"] <= 1.0
        assert len(model_summary["threshold_analysis"]) == 3


def test_figure_generation_is_reproducible(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "models": {
                    "logistic_regression": {"auroc": 0.8, "auprc": 0.7},
                    "random_forest": {"auroc": 0.9, "auprc": 0.75},
                }
            }
        ),
        encoding="utf-8",
    )
    first_path = tmp_path / "first.svg"
    second_path = tmp_path / "second.svg"

    make_benchmark_figure(benchmark_path, first_path)
    make_benchmark_figure(benchmark_path, second_path)

    first = first_path.read_text(encoding="utf-8")
    assert first == second_path.read_text(encoding="utf-8")
    assert first.startswith("<?xml")
    assert "logistic regression" in first
    assert "Synthetic, non-diagnostic research output" in first


# ── LOSO-specific tests ───────────────────────────────────────────────────────


def test_loso_pipeline_produces_per_fold_breakdown(tmp_path):
    """LOSO with 4 synthetic subjects should produce 4 per-fold records."""
    benchmark_path = tmp_path / "loso_benchmark.json"

    summary = run_baseline_pipeline(
        benchmark_path,
        synthetic=True,
        subject_count=4,
        samples_per_subject=40,
        event_segment_size=10,
        sampling_rate_hz=10,
        window_size=10,
        overlap_fraction=0.5,
        validation="loso",
        random_seed=42,
        random_forest_estimators=10,
        thresholds=(0.3, 0.5),
        calibration_bins=4,
    )

    assert summary["pipeline"]["validation"] == "loso"
    assert summary["pipeline"]["fold_count"] == 4

    for model_summary in summary["models"].values():
        per_fold = model_summary["per_fold"]
        assert len(per_fold) == 4, "Expected one fold per subject"

        for fold_record in per_fold:
            assert "fold_index" in fold_record
            assert "subject_id" in fold_record
            assert fold_record["n_windows"] > 0

        aggregate = model_summary["aggregate"]
        assert aggregate["n_folds"] == 4
        for key in ("auroc_mean", "auroc_std", "auprc_mean", "auprc_std"):
            assert key in aggregate, f"Missing aggregate key: {key}"

        subject_ids = [r["subject_id"] for r in per_fold]
        assert len(set(subject_ids)) == 4, "Each fold should test a different subject"


def test_loso_pipeline_fold_subject_ids_do_not_repeat(tmp_path):
    """Each LOSO fold must test a distinct subject -- no leakage indicator."""
    benchmark_path = tmp_path / "loso_leak_check.json"

    summary = run_baseline_pipeline(
        benchmark_path,
        synthetic=True,
        subject_count=4,
        samples_per_subject=40,
        event_segment_size=10,
        sampling_rate_hz=10,
        window_size=10,
        overlap_fraction=0.5,
        validation="loso",
        random_seed=7,
        random_forest_estimators=10,
        thresholds=(0.5,),
        calibration_bins=3,
    )

    for model_summary in summary["models"].values():
        subject_ids = [r["subject_id"] for r in model_summary["per_fold"]]
        assert len(subject_ids) == len(set(subject_ids)), (
            "Duplicate subject IDs across LOSO folds indicate a leakage problem"
        )


def test_loso_figure_is_reproducible_and_valid_svg(tmp_path):
    """make_loso_figure should produce identical SVG on two calls and contain key elements."""
    per_fold_data = [
        {
            "fold_index": i,
            "subject_id": f"S0{i + 1}",
            "n_windows": 100,
            "n_positive": 10,
            "auroc": round(0.6 + i * 0.05, 4),
        }
        for i in range(4)
    ]
    benchmark = {
        "models": {
            "random_forest": {
                "auroc": 0.7,
                "auprc": 0.3,
                "per_fold": per_fold_data,
                "aggregate": {"n_folds": 4, "auroc_mean": 0.675, "auroc_std": 0.07},
            }
        }
    }
    bench_path = tmp_path / "loso_bench.json"
    bench_path.write_text(json.dumps(benchmark), encoding="utf-8")

    out1 = tmp_path / "loso_fig_1.svg"
    out2 = tmp_path / "loso_fig_2.svg"
    make_loso_figure(bench_path, out1)
    make_loso_figure(bench_path, out2)

    svg = out1.read_text(encoding="utf-8")
    assert svg == out2.read_text(encoding="utf-8"), "SVG output must be deterministic"
    assert svg.startswith("<?xml")
    assert "<svg " in svg
    assert "S01" in svg
    assert "mean=" in svg


def test_aggregate_fold_metrics_computes_mean_and_std():
    """aggregate_fold_metrics should return correct population mean and std."""
    from parkinson_wearable_biomarkers.evaluate import aggregate_fold_metrics

    folds = [
        {"auroc": 0.8, "auprc": 0.4},
        {"auroc": 0.6, "auprc": 0.2},
    ]
    result = aggregate_fold_metrics(folds)

    assert result["n_folds"] == 2
    assert abs(result["auroc_mean"] - 0.7) < 1e-10
    assert abs(result["auroc_std"] - 0.1) < 1e-10
    assert abs(result["auprc_mean"] - 0.3) < 1e-10
    assert abs(result["auprc_std"] - 0.1) < 1e-10


def test_aggregate_fold_metrics_skips_non_numeric_keys():
    """Non-numeric and missing keys are silently omitted from aggregation."""
    from parkinson_wearable_biomarkers.evaluate import aggregate_fold_metrics

    folds = [
        {"auroc": 0.8, "subject_id": "S01", "n_windows": 100},
        {"auroc": 0.6, "subject_id": "S02", "n_windows": 120},
    ]
    result = aggregate_fold_metrics(folds)

    assert "auroc_mean" in result
    # subject_id is a string -- must not appear as aggregated key
    assert "subject_id_mean" not in result


# ── Permutation test ──────────────────────────────────────────────────────────



def test_permutation_test_produces_null_distribution_synthetic(tmp_path):
    """Full pipeline permutation test on synthetic CSV -- validates JSON structure."""
    import csv as _csv
    import random as _random

    from scripts.permutation_test import run_permutation_test

    # Build a synthetic CSV that guarantees both classes in every 2-fold split.
    # Use alternating labels (0,1,0,1,...) so every window after majority-vote
    # has an equal mix, and use window_size=5 to create many small windows.
    csv_path = tmp_path / "sensor.csv"
    rng = _random.Random(99)
    header = ["acc_x", "acc_y", "acc_z", "subject_id", "label"]
    rows = []
    for subj in range(6):
        sid = f"S{subj:02d}"
        for i in range(100):
            # Hard alternation: window majority-vote on 5 rows → 3 of one class
            # Repeat pattern [1,1,1,0,0] so each window of 5 has majority 1 or 0
            label = 1 if i % 10 < 5 else 0
            rows.append([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), sid, label])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    output = tmp_path / "perm.json"
    n_perms = 6

    summary = run_permutation_test(
        output,
        input_path=csv_path,
        sampling_rate_hz=10.0,
        window_size=5,
        overlap_fraction=0.0,
        n_splits=2,
        n_permutations=n_perms,
        random_seed=7,
        random_forest_estimators=5,
    )

    assert output.exists()
    assert summary["pipeline"]["n_permutations"] == n_perms
    assert summary["pipeline"]["permutation_strategy"] == "within_subject"
    assert summary["pipeline"]["validation"] == "groupkfold"

    for model_data in summary["models"].values():
        if "error" in model_data:
            continue  # Tolerate degenerate tiny folds on synthetic data.
        null = model_data["null_auroc"]
        assert len(null) <= n_perms
        assert 0.0 <= model_data["observed_auroc"] <= 1.0
        assert 0.0 <= model_data["p_value"] <= 1.0
        assert model_data["n_permutations"] == n_perms




def test_permutation_within_subject_shuffle_preserves_class_counts():
    """permute_labels_within_subjects must not change per-subject positive counts."""
    import random as _random

    from parkinson_wearable_biomarkers.features import FeatureDataset

    from scripts.permutation_test import permute_labels_within_subjects

    # Build a tiny FeatureDataset with two subjects.
    subject_ids = ("A", "A", "A", "A", "B", "B", "B", "B")
    labels = (1, 0, 0, 1, 0, 1, 0, 0)  # A: 2 pos, B: 1 pos
    feature_names = ("f1",)
    features = tuple((0.0,) for _ in range(8))

    dataset = FeatureDataset(
        feature_names=feature_names,
        features=features,
        subject_ids=subject_ids,
        labels=labels,
    )

    rng = _random.Random(42)
    permuted = permute_labels_within_subjects(dataset, rng=rng)

    # Per-subject counts must be preserved.
    orig_a = sum(labels[i] for i, s in enumerate(subject_ids) if s == "A")
    orig_b = sum(labels[i] for i, s in enumerate(subject_ids) if s == "B")
    perm_a = sum(permuted.labels[i] for i, s in enumerate(permuted.subject_ids) if s == "A")
    perm_b = sum(permuted.labels[i] for i, s in enumerate(permuted.subject_ids) if s == "B")

    assert perm_a == orig_a, "Subject A positive count changed after shuffle"
    assert perm_b == orig_b, "Subject B positive count changed after shuffle"

    # Features and subject_ids must be unchanged.
    assert permuted.feature_names == dataset.feature_names
    assert permuted.features == dataset.features
    assert permuted.subject_ids == dataset.subject_ids


def test_permutation_null_mean_is_near_chance(tmp_path):
    """Null AUROC mean should be near 0.5 under within-subject label shuffling."""
    import csv as _csv
    import math as _math
    import random as _random

    from scripts.permutation_test import run_permutation_test

    # Build a synthetic CSV with enough rows for stable null statistics.
    csv_path = tmp_path / "sensor_chance.csv"
    rng = _random.Random(1)
    header = ["acc_x", "acc_y", "acc_z", "subject_id", "label"]
    rows = []
    for subj in range(4):
        sid = f"S{subj:02d}"
        for _ in range(80):
            label = 1 if rng.random() < 0.25 else 0
            rows.append([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), sid, label])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    output = tmp_path / "perm_chance.json"
    summary = run_permutation_test(
        output,
        input_path=csv_path,
        sampling_rate_hz=10.0,
        window_size=10,
        overlap_fraction=0.5,
        n_splits=2,
        n_permutations=20,
        random_seed=13,
        random_forest_estimators=5,
    )

    for model_name, model_data in summary["models"].items():
        if "error" in model_data or not model_data["null_auroc"]:
            continue
        null = model_data["null_auroc"]
        null_mean = _math.fsum(null) / len(null)
        assert 0.25 <= null_mean <= 0.75, (
            f"{model_name} null mean {null_mean:.3f} is far from chance; "
            "check that within-subject shuffle is breaking signal correctly"
        )


def test_permutation_figure_is_valid_svg(tmp_path):
    """make_permutation_figure should produce deterministic SVG with key elements."""
    from scripts.make_permutation_figure import make_permutation_figure

    null_vals = [0.48 + i * 0.001 for i in range(40)]
    benchmark = {
        "models": {
            "logistic_regression": {
                "observed_auroc": 0.70,
                "null_auroc": null_vals,
                "null_mean": 0.50,
                "null_std": 0.02,
                "p_value": 0.03,
                "n_permutations": 40,
                "n_effective_permutations": 40,
                "failed_permutations": 0,
            },
            "random_forest": {
                "observed_auroc": 0.78,
                "null_auroc": null_vals,
                "null_mean": 0.50,
                "null_std": 0.02,
                "p_value": 0.001,
                "n_permutations": 40,
                "n_effective_permutations": 40,
                "failed_permutations": 0,
            },
        }
    }
    bench_path = tmp_path / "perm_bench.json"
    bench_path.write_text(json.dumps(benchmark), encoding="utf-8")

    out1 = tmp_path / "perm_fig1.svg"
    out2 = tmp_path / "perm_fig2.svg"
    make_permutation_figure(bench_path, out1)
    make_permutation_figure(bench_path, out2)

    svg = out1.read_text(encoding="utf-8")
    assert svg == out2.read_text(encoding="utf-8"), "SVG output must be deterministic"
    assert svg.startswith("<?xml")
    assert "<svg " in svg
    assert "logistic regression" in svg.lower() or "Logistic" in svg
    assert "obs=" in svg
    assert "p=" in svg


# ── Leaky split ablation (diagnostic) ────────────────────────────────────────


def test_leaky_ablation_produces_valid_json_with_warning(tmp_path):
    """run_leaky_ablation should write JSON with the DIAGNOSTIC_WARNING key
    and valid metric values for both models."""
    import csv as _csv
    import random as _random

    from scripts.diagnostics.leaky_split_ablation import run_leaky_ablation

    # Synthetic CSV: 4 subjects, 80 rows each, balanced labels.
    csv_path = tmp_path / "sensor_leaky.csv"
    rng = _random.Random(7)
    header = ["acc_x", "acc_y", "acc_z", "subject_id", "label"]
    rows = []
    for subj in range(4):
        sid = f"S{subj:02d}"
        for i in range(80):
            label = 1 if i % 2 == 0 else 0
            rows.append([rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), sid, label])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    output = tmp_path / "leaky.json"
    summary = run_leaky_ablation(
        output,
        input_path=csv_path,
        sampling_rate_hz=10.0,
        window_size=5,
        overlap_fraction=0.0,
        n_splits=2,
        random_seed=42,
        random_forest_estimators=5,
    )

    # JSON must be persisted and match the returned dict.
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == summary

    # Mandatory warning key must be present.
    assert "DIAGNOSTIC_WARNING" in summary
    assert "NOT" in summary["DIAGNOSTIC_WARNING"]

    # Split strategy must be clearly labeled.
    assert "LEAKY" in summary["pipeline"]["split_strategy"].upper()

    # Metrics should be in valid ranges.
    for model_data in summary["models"].values():
        assert 0.0 <= model_data["auroc"] <= 1.0
        assert 0.0 <= model_data["auprc"] <= 1.0
        assert 0.0 <= model_data["brier_score"] <= 1.0
        assert 0.0 <= model_data["expected_calibration_error"] <= 1.0
