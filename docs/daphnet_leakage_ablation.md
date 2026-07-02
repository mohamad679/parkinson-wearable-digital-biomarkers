# Leaky vs. Honest Split Ablation — Subject-Leakage Demonstration

> **Research-use boundary:** This is a preliminary, research-only
> reproducibility report. It is non-diagnostic, has not been clinically
> validated, and is not evidence of clinical readiness, safety, or utility for
> decisions about an individual.

> [!CAUTION]
> **The "Leaky" rows in this document are shown only to demonstrate the risk of
> naive splitting. They must never be cited as this project's benchmark result.**

## Purpose

The project README warns that randomly splitting windows into train and test
sets — ignoring which subject they came from — inflates AUROC substantially.
This document turns that assertion into a measured number.

The two strategies compared here use exactly the same feature matrix, the same
two models, the same class weights, and the same four metrics:

| Strategy | Fold assignment rule |
|---|---|
| **Honest (subject-aware)** | Entire subjects assigned to folds via GroupKFold; a subject's windows appear in training **or** test, never both |
| **Leaky (random)** | StratifiedKFold shuffles all windows globally, ignoring subject identity; windows from the same recording session can appear in both train and test |

## Mechanism of Leakage

Windows from the same recording session share:

- **Physiology:** the same person's gait kinematics, tremor amplitude, step
  cadence, and freeze severity.
- **Sensor placement:** identical device attachment angle, height, and tightness.
- **Signal artefacts:** subject-specific motion artefacts, cable noise, and
  calibration drift.
- **Recording conditions:** same walking environment, footwear, medication
  state, and time of day.

When the model trains on windows from subject *S* and is then tested on other
windows from the same subject *S*, it has already learned that subject's
fingerprint. The resulting AUROC does not reflect whether the model generalises
to a person it has never seen — it measures how well the model memorises
within-subject patterns. That is a fundamentally different (and much easier)
task.

## Comparison Table

| Model | Split | AUROC | AUPRC | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic regression | Honest (subject-aware GroupKFold) | 0.7113 | 0.2412 | 0.2127 | 0.3171 |
| Logistic regression | **Leaky (random StratifiedKFold)** | **0.7642** | **0.2486** | **0.2088** | **0.3275** |
| Logistic regression | Inflation (Leaky − Honest) | **Δ +0.0529** | **Δ +0.0074** | — | — |
| Random forest | Honest (subject-aware GroupKFold) | 0.7795 | 0.2280 | 0.0841 | 0.0333 |
| Random forest | **Leaky (random StratifiedKFold)** | **0.9470** | **0.6887** | **0.0480** | **0.0158** |
| Random forest | Inflation (Leaky − Honest) | **Δ +0.1675** | **Δ +0.4607** | — | — |

_Honest numbers: 3-fold GroupKFold from `results/daphnet_trunk_benchmark.json`.
Leaky numbers: StratifiedKFold from `results/daphnet_trunk_leaky_benchmark.json`
(not committed — see Reproduction below)._

## Interpretation

RF's leaky AUROC of 0.9470 is 0.1675 AUROC points above the honest estimate of
0.7795. Logistic regression shows a smaller but still meaningful pattern
(0.7642 vs 0.7113, Δ = +0.0529). The AUPRC inflation is even more dramatic for
RF: +0.4607 absolute (0.6887 vs 0.2280), because AUPRC is particularly
sensitive to memorisation of the rare positive windows.

The inflation arises because subject-specific gait signatures — physiology,
sensor placement, movement artefacts, and recording conditions — are the same
within a person across all their windows. A model that trains on some of
subject S's windows can exploit those signatures when tested on other windows
from the same subject, even if the labels are noisy. This is a fundamentally
easier task than generalising to a new person the model has never seen.

These leaky numbers are presented only to make the magnitude of the bias
concrete. They cannot be interpreted as evidence of clinical utility, and
any researcher who uses a subject-unaware split on this or a similar dataset
will produce results that are not reproducible in a new participant.

> **The leaky numbers above are shown only to demonstrate the risk of naive
> splitting and must never be cited as this project's benchmark result.**

## Limitations of This Comparison

- The honest numbers above use 3-fold GroupKFold, not the 10-fold LOSO
  reported in [daphnet_loso_report.md](daphnet_loso_report.md). LOSO is
  the primary result; this comparison uses 3-fold to keep fold assignments
  identical between the two strategies.
- Subject count is N=10, so the honest per-fold variance is already high.
  The measured inflation Δ is dataset- and configuration-specific.
- No hyperparameter tuning was performed in either strategy; with tuning,
  leakage effects would typically be larger still.
- Brier score and ECE comparisons may show unexpected directions because
  probability calibration is a different axis of performance from AUROC/AUPRC.

## Local Reproduction Workflow

Run the honest 3-fold benchmark (if not already present):

```bash
python scripts/run_baselines.py \
  --input data/processed/daphnet_trunk.csv \
  --output results/daphnet_trunk_benchmark.json \
  --sampling-rate 64 --window-size 128 --overlap 0.5 \
  --folds 3 --validation groupkfold \
  --thresholds 0.3 0.5 0.7 --calibration-bins 5 --random-seed 42
```

Run the leaky diagnostic (same parameters, different split):

```bash
python scripts/diagnostics/leaky_split_ablation.py \
  --input data/processed/daphnet_trunk.csv \
  --output results/daphnet_trunk_leaky_benchmark.json \
  --sampling-rate 64 --window-size 128 --overlap 0.5 \
  --folds 3 --random-seed 42
```

After running, fill in the Comparison Table from the `auroc`, `auprc`,
`brier_score`, and `expected_calibration_error` fields in each JSON, then
compute Δ = Leaky − Honest and replace the Interpretation paragraph template.

The leaky benchmark JSON must remain uncommitted.
