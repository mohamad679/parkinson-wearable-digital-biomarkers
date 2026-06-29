# Parkinson Wearable Digital Biomarkers

A reproducible research baseline for detecting freezing-of-gait (FoG) events from
wearable accelerometer time-series. The project connects validated CSV loading,
subject-safe windowing, interpretable signal features, subject-aware validation,
class-weighted baseline models, discrimination metrics, and probability calibration.

> **Research-use boundary:** This repository is not a diagnostic medical device and
> is not clinically validated. Its outputs must not be used to diagnose Parkinson's
> disease, direct treatment, determine fall risk, or make decisions about an individual.

## Motivation and Scope

Wearable accelerometers can support research into movement patterns that may be useful
as candidate digital biomarkers. A credible baseline must evaluate whether a signal
generalizes to people who were not used for model fitting, report performance under
class imbalance, and examine whether predicted probabilities are calibrated.

This repository provides a deterministic, command-line-reproducible implementation of
that baseline. It prioritizes interpretable logistic regression and random forest models
over deep learning. Synthetic data is included only as a pipeline check; no raw Kaggle,
clinical, or participant data is bundled.

## Pipeline

1. Validate configurable accelerometer, subject-ID, and binary-label CSV columns.
2. Create fixed-size overlapping windows without crossing subject boundaries.
3. Assign mixed-label windows by majority label, breaking ties with the first label in
   the window.
4. Extract per-axis and magnitude statistics, energy, dominant frequency, and rolling
   variance with finite handling for missing signal values.
5. Generate deterministic GroupKFold-style or Leave-One-Subject-Out splits.
6. Fit class-weighted logistic regression and random forest baselines.
7. Report AUROC, AUPRC, threshold metrics, Brier score, Expected Calibration Error
   (ECE), and calibration-curve data.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pinned requirements include scikit-learn for the two baseline estimators. The core
data, windowing, feature, metric, and calibration utilities otherwise use the Python
standard library.

## Command-Line Workflow

Generated CSV, JSON, and SVG files are reproducibility artifacts. Keep them outside
version control and never commit participant or processed sensor data.

### Run tests and quality checks

```bash
python -m pytest
python -m ruff check .
python scripts/smoke_test.py
```

The smoke test runs the complete synthetic command-line workflow in a temporary
directory and leaves no generated data in the repository.

### Generate deterministic synthetic data

```bash
mkdir -p artifacts
python scripts/prepare_data.py \
  --synthetic \
  --output artifacts/synthetic_sensor_data.csv \
  --subjects 6 \
  --samples-per-subject 80 \
  --sampling-rate 20 \
  --event-segment-size 20 \
  --random-seed 42
```

This data is artificial and intended only for tests, examples, and pipeline validation.
It does not represent clinical performance or the variability of real wearable data.

### Run subject-aware baselines

Run from the prepared synthetic CSV:

```bash
python scripts/run_baselines.py \
  --input artifacts/synthetic_sensor_data.csv \
  --output artifacts/synthetic_benchmark.json \
  --sampling-rate 20 \
  --window-size 20 \
  --overlap 0.5 \
  --folds 3 \
  --thresholds 0.3 0.5 0.7 \
  --calibration-bins 5 \
  --random-seed 42
```

Or let the baseline command generate temporary synthetic input itself:

```bash
python scripts/run_baselines.py \
  --synthetic \
  --output artifacts/synthetic_benchmark.json
```

`--window-size` is a number of samples, not seconds. The benchmark JSON contains
pipeline metadata and out-of-fold metrics for both baseline models. The values produced
from synthetic data are software checks, not estimates of real-world or clinical utility.

### Generate a reproducible figure

```bash
python scripts/make_figures.py \
  --input artifacts/synthetic_benchmark.json \
  --output artifacts/synthetic_benchmark.svg
```

The figure is a dependency-free SVG comparison of AUROC and AUPRC. It is labeled as
synthetic, non-diagnostic research output.

Use `python scripts/<script>.py --help` for all available options. The values in
`configs/baseline.yaml` document baseline defaults for the broader project; the current
CLI receives its settings through explicit arguments and does not automatically load
that YAML file.

## Subject-Aware Validation

The main validation strategy is deterministic GroupKFold-style splitting by subject.
Entire subjects are assigned to test folds, with groups balanced approximately by row
count. Leave-One-Subject-Out splitting is also available through the Python API. Every
generated fold is checked explicitly for shared train/test subject IDs.

Subject-aware validation matters because windows from one person share physiology,
wearing style, sensor placement, gait characteristics, and recording conditions. A
model can exploit those person-specific patterns without learning a signal that
generalizes to a new person.

> **Subject-leakage warning:** Never randomly split windows from the same subjects into
> both training and test sets for the main evaluation. That design leaks subject-specific
> information and can substantially overstate generalization. Keep preprocessing,
> feature selection, threshold selection, and calibration fitting inside the training
> side of each subject-aware fold when extending this baseline.

## Metrics, Imbalance, and Calibration

FoG event windows may be much less frequent than non-event windows. Accuracy can appear
high when a model mostly predicts the majority class, so it is not the primary metric in
this project.

- **AUPRC** emphasizes positive-event retrieval and precision. Its baseline depends on
  positive prevalence, so every report should include class counts and prevalence.
- **AUROC** measures ranking across both classes but can look optimistic when negatives
  dominate; it should be interpreted alongside AUPRC.
- **F1, sensitivity, specificity, precision, and confusion counts** describe behavior at
  explicit thresholds. Thresholds must be selected without using test-fold outcomes.
- **Brier score, ECE, and calibration curves** assess whether probability values match
  observed event frequencies. Good discrimination does not guarantee reliable
  probabilities.

Calibration estimates can be unstable with small samples or sparse bins. They are
descriptive research outputs, not individualized risk estimates.

## Relevance to Digital Biomarkers for Parkinson’s Disease

This project tests an engineering question: whether wearable accelerometer signals can
support reproducible, calibrated, subject-generalizing candidate measures related to
FoG events. Subject-aware validation, explicit class-imbalance reporting, and calibration
are prerequisites for evaluating such candidates responsibly.

Even strong benchmark results would not by themselves establish a validated digital
biomarker. Clinical relevance would require representative prospective data, reliable
reference labels, protocol and device robustness, external validation, subgroup
analysis, longitudinal reliability, and assessment in the intended context of use.

## Repository Structure

```text
.
├── configs/
│   └── baseline.yaml                 # Reference baseline settings
├── scripts/
│   ├── prepare_data.py               # Deterministic synthetic CSV generator
│   ├── run_baselines.py              # End-to-end subject-aware benchmark
│   ├── make_figures.py               # Reproducible benchmark SVG
│   └── smoke_test.py                 # Temporary end-to-end CLI check
├── src/parkinson_wearable_biomarkers/
│   ├── data.py                       # CSV schema and loading
│   ├── preprocessing.py              # Subject-safe windowing
│   ├── features.py                   # Interpretable signal features
│   ├── validation.py                 # GroupKFold-style and LOSO splits
│   ├── models.py                     # Class-weighted baselines
│   ├── evaluate.py                   # Discrimination and threshold metrics
│   └── calibration.py                # Brier, ECE, and calibration curves
├── tests/                             # Synthetic unit and integration tests
├── results/
│   └── benchmark_report.md           # Reusable reporting template
├── MODEL_CARD.md                     # Intended use and model limitations
└── requirements.txt                  # Pinned development/runtime dependencies
```

## Limitations

- No real participant data, trained production artifact, or clinical benchmark is
  distributed with this repository.
- Synthetic data verifies software behavior but does not reproduce disease, FoG,
  comorbidities, medication effects, daily-living contexts, or sensor artifacts.
- Window-level majority labels can blur short events and depend on window length and
  overlap.
- Handcrafted features and two classical models are deliberately limited baselines.
- Grouped cross-validation does not replace external, prospective, multi-site, or
  longitudinal validation.
- Metrics can be unstable with few subjects, rare positives, label noise, or sparse
  calibration bins.
- Device placement, sampling rate, preprocessing, missingness, demographics, disease
  stage, and recording protocol may create dataset shift.
- No fairness, subgroup robustness, uncertainty, causal validity, usability, or safety
  claims are established.
- Probability outputs are model scores and must not be interpreted as diagnosis,
  prognosis, or patient-specific clinical risk.

## Reporting

Use [MODEL_CARD.md](MODEL_CARD.md) to review intended use and ethical constraints. Copy
[results/benchmark_report.md](results/benchmark_report.md) for each experiment and
replace every placeholder with traceable data, configuration, fold, and artifact details.
