import json

from scripts.make_figures import make_benchmark_figure
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
