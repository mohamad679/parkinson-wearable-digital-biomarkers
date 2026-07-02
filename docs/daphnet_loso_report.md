# Daphnet Trunk-Sensor LOSO Benchmark

> **Primary result:** This report supersedes the 3-fold grouped estimate in
> [daphnet_real_data_report.md](daphnet_real_data_report.md) as the primary
> validation result for the Daphnet trunk benchmark.

> **Research-use boundary:** This is a preliminary, research-only
> reproducibility report. It is non-diagnostic, has not been clinically
> validated, and is not evidence of clinical readiness, safety, or utility for
> decisions about an individual.

## Dataset Provenance and Citation

The local benchmark used the Daphnet Freezing of Gait dataset after manual
download. Raw recordings, the source ZIP, the converted CSV, and the benchmark
JSON are not committed to this repository. The publishable artifact retained
here contains only aggregate metrics.

The Daphnet README requires citation of:

Marc Bächlin, Meir Plotnik, Daniel Roggen, Inbal Maidan, Jeffrey M. Hausdorff,
Nir Giladi, and Gerhard Tröster, *Wearable Assistant for Parkinson's Disease
Patients With the Freezing of Gait Symptom*. IEEE Transactions on Information
Technology in Biomedicine, 14(2), March 2010, pages 436–446.

Users must review and comply with the dataset's current terms, documentation,
and citation requirements before obtaining or using it.

## Why LOSO instead of GroupKFold?

Leave-One-Subject-Out (LOSO) validation holds out exactly one subject per fold,
giving a per-subject breakdown of model performance and an honest simulation of
generalisation to a new individual. The previous 3-fold grouped estimate pooled
3–4 subjects into each test fold, masking subject-to-subject variance and making
the aggregate AUROC harder to interpret.

LOSO produces 10 folds for 10 subjects, allowing mean ± standard deviation to
be reported across folds.

> **Caveat:** N=10 subjects; standard deviation is reported but should not be
> read as a stable population estimate. Inter-subject variance in a cohort of
> this size reflects both model sensitivity and the heterogeneity of individual
> freezing patterns.

## Exact Local Benchmark Configuration

| Setting | Value |
| --- | ---: |
| Sensor | trunk/hip |
| Sampling rate | 64.0 Hz |
| Window size | 128 samples |
| Approximate window duration | 2.0 seconds |
| Overlap fraction | 0.5 |
| Validation | Leave-One-Subject-Out (LOSO) |
| Subjects | 10 |
| Total complete windows | 17,815 |
| Feature count | 25 |
| Random seed | 42 |

## Per-Subject Metrics — Random Forest

<!-- PLACEHOLDER: run the pipeline (see Reproduction section) and fill in these
     values before committing. Remove this comment when real numbers are added. -->

| Subject | n_windows | n_positive | AUROC | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 | … | … | … | … | … | … |
| S02 | … | … | … | … | … | … |
| S03 | … | … | … | … | … | … |
| S04 | … | … | … | … | … | … |
| S05 | … | … | … | … | … | … |
| S06 | … | … | … | … | … | … |
| S07 | … | … | … | … | … | … |
| S08 | … | … | … | … | … | … |
| S09 | … | … | … | … | … | … |
| S10 | … | … | … | … | … | … |
| **Mean ± SD** | | | **… ± …** | **… ± …** | **… ± …** | **… ± …** |

## Per-Subject Metrics — Logistic Regression

| Subject | n_windows | n_positive | AUROC | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S01 | … | … | … | … | … | … |
| S02 | … | … | … | … | … | … |
| S03 | … | … | … | … | … | … |
| S04 | … | … | … | … | … | … |
| S05 | … | … | … | … | … | … |
| S06 | … | … | … | … | … | … |
| S07 | … | … | … | … | … | … |
| S08 | … | … | … | … | … | … |
| S09 | … | … | … | … | … | … |
| S10 | … | … | … | … | … | … |
| **Mean ± SD** | | | **… ± …** | **… ± …** | **… ± …** | **… ± …** |

## Per-Subject AUROC Figure

![Per-subject AUROC dot plot — random forest](figures/daphnet_loso_benchmark.svg)

The figure shows one dot per subject on the y-axis (AUROC) with a horizontal
dashed line at the cross-subject mean. It contains only aggregate discrimination
metrics and no participant-level or raw sensor information.

> **Note:** The SVG above is a generated artifact produced by
> `scripts/make_loso_figure.py` and is not committed (covered by `.gitignore`).
> Run the reproduction workflow below to regenerate it.

## Interpretation

The LOSO breakdown exposes subject-to-subject variance that the pooled 3-fold
estimate concealed. Some subjects may be substantially harder to classify than
others, reflecting differences in freeze severity, gait style, or sensor
placement.

These results are a reproducibility benchmark for two simple models on one
trunk-only configuration. They are not clinical evidence. They do not establish
diagnostic validity, reliable real-time FoG detection, patient benefit, safety,
or suitability for clinical decision-making.

## Limitations

- Only the trunk/hip sensor was used; ankle, thigh, and multi-sensor
  combinations were not evaluated.
- Features were simple handcrafted window summaries rather than a comprehensive
  signal representation.
- No nested hyperparameter tuning was performed.
- LOSO standard deviation across N=10 subjects is descriptive only and should
  not be extrapolated to population-level variance.
- No external dataset or prospective validation was performed.
- No subgroup, fairness, device-shift, medication-state, or protocol analysis.
- Window labels can obscure short transitions; overlapping windows are
  correlated within subjects.
- Reference annotations may not reflect unscripted daily living conditions.
- No clinical-readiness claim is made.

## Local Reproduction Workflow

After manually obtaining Daphnet, keep the archive under the ignored `data/raw/`
path and run:

```bash
python scripts/prepare_daphnet.py \
  --input data/raw/daphnet.zip \
  --output data/processed/daphnet_trunk.csv \
  --sensor trunk

python scripts/run_baselines.py \
  --input data/processed/daphnet_trunk.csv \
  --output results/daphnet_trunk_loso_benchmark.json \
  --sampling-rate 64 \
  --window-size 128 \
  --overlap 0.5 \
  --validation loso \
  --thresholds 0.3 0.5 0.7 \
  --calibration-bins 5 \
  --random-seed 42

python scripts/make_loso_figure.py \
  --input results/daphnet_trunk_loso_benchmark.json \
  --output docs/figures/daphnet_loso_benchmark.svg
```

After running, fill in the per-subject tables above from the `per_fold` arrays
in `results/daphnet_trunk_loso_benchmark.json`. The `aggregate` section in the
JSON contains `auroc_mean`, `auroc_std`, etc. for the Mean ± SD row.

The raw archive, converted CSV, benchmark JSON, and generated
`docs/figures/daphnet_loso_benchmark.svg` are ignored and must remain
uncommitted.
