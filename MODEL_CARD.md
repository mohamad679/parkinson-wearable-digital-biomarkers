# Model Card: Parkinson Wearable FoG Research Baselines

## Model Details

This repository provides two deterministic binary-classification baselines:

- logistic regression with balanced class weights; and
- random forest with balanced class weights and single-threaded fitting.

Both models consume the finite feature matrix produced from fixed-size accelerometer
windows and output a probability score for positive label `1`. Seeds and estimator
settings must be recorded for reproducibility. No pretrained model artifact is included.

## Intended Use

The baselines are intended for:

- reproducible methods research on wearable accelerometer time-series;
- evaluation of candidate FoG-related digital-biomarker signals;
- comparison of interpretable feature sets and subject-aware validation designs;
- study of class imbalance, operating thresholds, and probability calibration; and
- software verification using deterministic synthetic toy data.

Any research use with participant data requires appropriate governance, consent,
privacy protection, security, and protocol-specific review.

## Out-of-Scope and Prohibited Uses

These models are not intended to:

- diagnose or screen for Parkinson's disease or any other condition;
- determine whether an individual is experiencing FoG in a clinical setting;
- guide medication, treatment, rehabilitation, supervision, or fall-prevention decisions;
- provide prognosis or patient-specific risk estimates;
- replace clinician assessment or validated reference annotations;
- support unsupervised deployment, real-time safety intervention, or regulatory claims; or
- claim clinical readiness from synthetic or retrospective benchmark results.

## Data

No raw Kaggle, clinical, participant, or processed sensor dataset is committed. Tests and
the command-line smoke workflow use deterministic synthetic accelerometer rows with
artificial binary labels. Those rows exist to exercise the code and are not a simulation
of Parkinson's disease or a representative FoG cohort.

For user-supplied CSV data, the schema supports configurable accelerometer columns, a
subject-ID column, and a binary label column. Data provenance, device, placement,
sampling rate, protocol, missingness, annotation method, inclusion criteria, consent,
and governance must be documented separately for every study.

## Inputs and Features

The pipeline creates fixed-size windows without crossing contiguous subject boundaries.
Incomplete trailing windows are discarded. A mixed-label window receives its majority
sample label; a tie is resolved by the first label in the window.

Features include per-axis and acceleration-magnitude mean, population standard deviation,
minimum, maximum, sum-of-squares energy, and dominant non-DC frequency. Mean rolling
variance is added for magnitude. Non-finite samples are handled deterministically so the
model matrix contains finite values.

## Validation Strategy

The required primary strategy is subject-aware GroupKFold-style validation or
Leave-One-Subject-Out validation. No subject ID may occur in both training and test rows
within a fold. The implementation checks folds explicitly for leakage.

Random window-level splitting is not an acceptable primary evaluation because it can
place highly related windows from one person on both sides of the split. Any feature
selection, hyperparameter tuning, threshold selection, resampling, or calibration fitting
added later must also be learned using training data only, preferably in a nested design.

At minimum, reports should include the number of subjects, positive/negative windows,
prevalence, fold construction, per-fold results, and a subject-leakage audit.

## Metrics

The implemented evaluation includes:

- AUROC and AUPRC;
- precision, F1, sensitivity, specificity, and confusion counts at configurable thresholds;
- positive support and prevalence context;
- Brier score;
- Expected Calibration Error; and
- equal-width calibration-curve data.

AUPRC is essential when positive FoG windows are rare, while AUROC should not be read in
isolation. Accuracy is not treated as the main metric. Calibration metrics are necessary
because ranking performance does not show whether probability values correspond to
observed frequencies. Threshold and calibration results must be reported with sample
counts and uncertainty where feasible.

## Limitations

- The repository contains research baselines, not optimized or clinically validated models.
- Synthetic tests cannot establish external validity or expected real-data performance.
- Reference labels may be noisy, subjective, delayed, or misaligned with sensor windows.
- Handcrafted features may miss relevant temporal structure and may encode device or
  protocol artifacts.
- Window length, overlap, label aggregation, sampling rate, and sensor placement can
  materially change results.
- Few subjects or rare events can produce high-variance discrimination and calibration
  estimates.
- Grouped cross-validation cannot demonstrate transportability across sites, devices,
  populations, disease stages, medication states, or daily-living conditions.
- Class weighting changes model fitting but does not solve representation gaps, label
  bias, dataset shift, or calibration drift.

## Ethical and Safety Cautions

False negatives could miss events of research interest; false positives could overstate
event burden. If model outputs were improperly treated as clinical facts, either error
could affect autonomy, anxiety, care decisions, or resource allocation. Aggregate results
may also hide worse performance for demographic, disease-stage, mobility, device, or
site subgroups.

Wearable recordings can reveal sensitive behavioral and health information. Studies
should minimize collection, restrict access, define retention, evaluate re-identification
risk, and communicate model uncertainty. Performance and calibration should be audited
across relevant subgroups before any consideration of higher-stakes use.

## Recommended Reporting

Use `results/benchmark_report.md` and record:

1. data provenance, governance, cohort, devices, and labeling;
2. exact preprocessing, windowing, feature, model, and seed settings;
3. subject-aware folds and leakage checks;
4. class counts and prevalence;
5. per-fold and aggregate discrimination, threshold, and calibration results;
6. uncertainty estimates and subgroup analyses where supported;
7. failures, missing data, exclusions, and deviations; and
8. the explicit non-diagnostic interpretation boundary.
