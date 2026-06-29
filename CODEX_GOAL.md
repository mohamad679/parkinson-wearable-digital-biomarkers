/goal Build the repository parkinson-wearable-digital-biomarkers as a research-grade, test-gated Python project for Parkinson's freezing-of-gait detection from wearable accelerometer time-series.

Follow AGENTS.md strictly.

Work milestone by milestone. Do not proceed to the next milestone unless all tests pass.

Current repository state:
- Git branch: agent/build-parkinson-wearable-pipeline
- Python package: src/parkinson_wearable_biomarkers
- Tests: tests/
- Quality gates:
  - python -m pytest
  - python -m ruff check .

Main scientific goal:
Build a reproducible baseline pipeline for detecting Parkinson's freezing-of-gait events from wearable accelerometer signals, including preprocessing, windowing, signal-derived features, subject-aware validation, class-imbalance analysis, calibration metrics, and cautious non-clinical digital-biomarker interpretation.

Required implementation milestones:

Milestone 1 — Project infrastructure
- Add GitHub Actions CI.
- Add a basic smoke test script.
- Add config file configs/baseline.yaml.
- Ensure python -m pytest and python -m ruff check . pass.

Milestone 2 — Data schema and loading
- Implement src/parkinson_wearable_biomarkers/data.py.
- Support CSV loading.
- Support configurable accelerometer columns.
- Support subject ID column.
- Support label column.
- Validate missing columns.
- Validate labels.
- Add unit tests.

Milestone 3 — Windowing
- Implement src/parkinson_wearable_biomarkers/preprocessing.py.
- Add configurable window size and overlap.
- Preserve subject IDs.
- Preserve labels.
- Add tests for window shape, overlap behavior, edge cases, and subject preservation.

Milestone 4 — Feature extraction
- Implement src/parkinson_wearable_biomarkers/features.py.
- Extract mean, std, min, max per axis.
- Extract acceleration magnitude features.
- Extract energy.
- Extract dominant frequency.
- Extract rolling variance if appropriate.
- Handle NaN safely.
- Add tests for output shape, feature names, deterministic behavior, and NaN handling.

Milestone 5 — Subject-aware validation
- Implement src/parkinson_wearable_biomarkers/validation.py.
- Add GroupKFold splitting.
- Add optional Leave-One-Subject-Out splitting.
- Add explicit subject-leakage checks.
- Add tests proving no subject appears in both train and test.

Milestone 6 — Metrics and calibration
- Implement src/parkinson_wearable_biomarkers/evaluate.py.
- Implement AUROC, AUPRC, F1, sensitivity, specificity, confusion matrix, threshold analysis.
- Implement src/parkinson_wearable_biomarkers/calibration.py.
- Implement Brier score, Expected Calibration Error, calibration curve data.
- Add tests on toy data.

Milestone 7 — Baseline models
- Implement src/parkinson_wearable_biomarkers/models.py.
- Add Logistic Regression with class weighting.
- Add Random Forest baseline.
- Use deterministic random seeds.
- Add tests where feasible.

Milestone 8 — Command-line scripts
- Implement scripts/prepare_data.py.
- Implement scripts/run_baselines.py.
- Implement scripts/make_figures.py.
- Scripts must run on synthetic toy data without raw Kaggle data.
- Add or update smoke test accordingly.

Milestone 9 — Documentation
- Expand README.md with project motivation, installation, commands, validation strategy, leakage warning, limitations, and relevance to digital biomarkers.
- Add MODEL_CARD.md.
- Add results/benchmark_report.md template.
- Keep all clinical claims cautious and non-diagnostic.

Hard rules:
- Do not commit raw data.
- Do not use random split as main validation.
- Demonstrate leakage risk only as a warning if random split is mentioned.
- Before every commit run:
  - python -m pytest
  - python -m ruff check .
- If any test fails, fix it before continuing.
- Commit after each completed milestone with a clear commit message.
- Do not push to main.
- Do not push anything until all milestones are complete and tests pass.
