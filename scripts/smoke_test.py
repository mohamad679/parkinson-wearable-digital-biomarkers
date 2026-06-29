"""Run the synthetic command-line pipeline as a repository smoke test."""

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Verify package import, configuration, commands, and generated artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    package = importlib.import_module("parkinson_wearable_biomarkers")
    config_path = project_root / "configs" / "baseline.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {config_path}")

    with tempfile.TemporaryDirectory(prefix="parkinson-smoke-") as directory:
        temporary_directory = Path(directory)
        csv_path = temporary_directory / "synthetic.csv"
        benchmark_path = temporary_directory / "benchmark.json"
        figure_path = temporary_directory / "benchmark.svg"
        _run_command(
            project_root,
            "scripts/prepare_data.py",
            "--synthetic",
            "--output",
            str(csv_path),
            "--subjects",
            "4",
            "--samples-per-subject",
            "40",
            "--sampling-rate",
            "10",
            "--event-segment-size",
            "10",
        )
        _run_command(
            project_root,
            "scripts/run_baselines.py",
            "--input",
            str(csv_path),
            "--output",
            str(benchmark_path),
            "--sampling-rate",
            "10",
            "--window-size",
            "10",
            "--folds",
            "2",
            "--random-forest-estimators",
            "10",
            "--calibration-bins",
            "4",
        )
        _run_command(
            project_root,
            "scripts/make_figures.py",
            "--input",
            str(benchmark_path),
            "--output",
            str(figure_path),
        )

        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if set(benchmark["models"]) != {"logistic_regression", "random_forest"}:
            raise RuntimeError("Smoke benchmark is missing baseline model results")
        if not figure_path.read_text(encoding="utf-8").startswith("<?xml"):
            raise RuntimeError("Smoke figure is not a valid generated SVG")

    package_name = package.__name__
    print(f"Smoke test passed for {package_name} using {config_path.name}.")


def _run_command(project_root: Path, script: str, *arguments: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed ({script}):\n{completed.stdout}\n{completed.stderr}"
        )


if __name__ == "__main__":
    main()
