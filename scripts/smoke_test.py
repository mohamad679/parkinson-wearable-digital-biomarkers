"""Run a minimal installation and configuration smoke test."""

import importlib
import sys
from pathlib import Path


def main() -> None:
    """Verify that the package imports and the baseline configuration exists."""
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    package = importlib.import_module("parkinson_wearable_biomarkers")
    config_path = project_root / "configs" / "baseline.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Baseline configuration not found: {config_path}")

    package_name = package.__name__
    print(f"Smoke test passed for {package_name} using {config_path.name}.")


if __name__ == "__main__":
    main()
