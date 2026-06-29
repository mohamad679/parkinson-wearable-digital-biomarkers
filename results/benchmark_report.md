# Benchmark Report Template

> **Required interpretation boundary:** This report describes a research benchmark. It
> does not establish diagnostic validity, clinical utility, safety, or readiness for use
> in decisions about an individual. Replace every `TBD` field before publishing results.

## 1. Report Metadata

- Experiment name: `TBD`
- Report date: `TBD`
- Repository commit: `TBD`
- Author/reviewer: `TBD`
- Execution environment: `TBD`
- Random seed(s): `TBD`
- Benchmark JSON path and checksum: `TBD`
- Figure path and checksum: `TBD`
- Configuration path or full CLI command: `TBD`

## 2. Research Question and Intended Context

- Research question: `TBD`
- Intended research use: `TBD`
- Positive-label definition: `TBD`
- Unit of analysis (sample/window/recording): `TBD`
- Explicit non-use statements: `TBD`

Do not frame the benchmark as diagnosis, prognosis, treatment guidance, fall-risk
determination, or validated patient monitoring.

## 3. Data Provenance and Governance

- Dataset name/version: `TBD`
- Synthetic or participant data: `TBD`
- Source and collection protocol: `TBD`
- Consent/approval/governance basis: `TBD`
- Device, placement, axes, and sampling rate: `TBD`
- Reference-label method and annotator information: `TBD`
- Inclusion/exclusion criteria: `TBD`
- Missing-data and quality-control rules: `TBD`
- Data-access and privacy controls: `TBD`

No raw or processed participant data should be committed to this repository.

## 4. Analysis Population

| Quantity | Value |
| --- | --- |
| Subjects | `TBD` |
| Recordings | `TBD` |
| Raw samples | `TBD` |
| Complete windows | `TBD` |
| Positive windows | `TBD` |
| Negative windows | `TBD` |
| Positive prevalence | `TBD` |
| Excluded rows/windows/subjects and reasons | `TBD` |

Describe whether subjects, labels, or recording conditions are unevenly distributed.

## 5. Pipeline Configuration

### Data schema

- Subject-ID column: `TBD`
- Label column: `TBD`
- Accelerometer columns: `TBD`
- Allowed labels and mapping: `TBD`

### Windowing

- Window size in samples and seconds: `TBD`
- Overlap fraction: `TBD`
- Mixed-label strategy: `TBD`
- Incomplete-window handling: `TBD`

### Features

- Feature names/version: `TBD`
- Missing/non-finite handling: `TBD`
- Sampling rate used for frequency features: `TBD`
- Rolling-variance window: `TBD`

### Models

- Logistic regression settings: `TBD`
- Random forest settings: `TBD`
- Class-weighting strategy: `TBD`
- Hyperparameter selection procedure: `TBD`

## 6. Subject-Aware Validation and Leakage Audit

- Strategy (GroupKFold or Leave-One-Subject-Out): `TBD`
- Number of folds: `TBD`
- Subject allocation per fold: `TBD`
- Were preprocessing and feature decisions fit using training data only? `TBD`
- Was threshold selection performed without test-fold outcomes? `TBD`
- Was calibration fit separately from final test outcomes, if applicable? `TBD`
- Explicit subject-overlap check result for every fold: `TBD`
- Nested validation or tuning design: `TBD`

> Random window-level splitting is not an acceptable replacement for the primary
> subject-aware evaluation. Any deviation must be labeled as exploratory and must not be
> used to claim generalization to unseen subjects.

## 7. Aggregate Results

Report confidence intervals or another uncertainty estimate when the data support it.
Do not enter synthetic smoke-test values as claims about real performance.

| Model | AUROC | AUPRC | Brier score | ECE | Positive prevalence | Uncertainty method |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic regression | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Random forest | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

Explain AUPRC relative to positive prevalence. Interpret AUROC and AUPRC together.

## 8. Threshold Analysis

Document how each threshold was selected. A threshold chosen using held-out test labels
invalidates an unbiased test estimate.

| Model | Threshold | Precision | F1 | Sensitivity | Specificity | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Random forest | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## 9. Per-Fold Results

| Fold | Test subjects | Positive/negative windows | Model | AUROC | AUPRC | Brier | ECE | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

Describe heterogeneity across folds rather than reporting only a pooled value.

## 10. Calibration

- Number and definition of bins: `TBD`
- Calibration fitting procedure, if any: `TBD`
- Sparse-bin warning or minimum count: `TBD`
- Brier score interpretation: `TBD`
- ECE interpretation and sensitivity to binning: `TBD`
- Calibration figure: `TBD`

Probability calibration is descriptive unless assessed on data independent of fitting
and calibration. It must not be interpreted as individualized clinical risk.

## 11. Subgroup, Device, and Robustness Checks

Only report categories supported by appropriate governance and adequate sample sizes.

| Analysis | Groups/conditions | Sample sizes | Result | Uncertainty | Caution |
| --- | --- | --- | --- | --- | --- |
| Demographic subgroup | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Device or placement | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Site or protocol | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| Missingness/noise stress test | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

## 12. Limitations and Failure Analysis

- Dataset representativeness: `TBD`
- Label quality and timing: `TBD`
- Subject count and event prevalence: `TBD`
- Device/protocol dependence: `TBD`
- Windowing and feature limitations: `TBD`
- Calibration uncertainty: `TBD`
- Observed failure cases: `TBD`
- Unassessed risks or subgroups: `TBD`

## 13. Reproducibility Record

```bash
# Environment installation
TBD

# Data preparation
TBD

# Baseline execution
TBD

# Figure generation
TBD

# Quality gates
python -m pytest
python -m ruff check .
python scripts/smoke_test.py
```

- Input manifest/checksums: `TBD`
- Output manifest/checksums: `TBD`
- Logs and warnings: `TBD`
- Deviations from the committed pipeline: `TBD`

## 14. Cautious Interpretation

### Supported conclusion

`TBD: State only what the subject-aware research evaluation directly supports.`

### Unsupported conclusions

- Diagnostic validity: **not established**
- Clinical utility or safety: **not established**
- Generalization beyond evaluated subjects/sites/devices: **not established**
- Individual prognosis or treatment relevance: **not established**

### Required next evidence

`TBD: List prospective, external, subgroup, longitudinal, calibration, and usability
evidence needed for the stated research context.`
