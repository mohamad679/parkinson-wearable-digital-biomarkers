# Daphnet Artifact Manifest

> **Research-use boundary:** This manifest documents local, ignored artifacts used
> to reproduce the Daphnet trunk-sensor benchmark. It is not clinical evidence and
> does not contain raw sensor rows or participant-level records.

## Repository State

| Item | Value |
| --- | --- |
| Audited branch | `agent/phase-1-2-3-validation-ablation` |
| Audited commit before final permutation update | `b01066cf636dfe1308302e320d63b57dd0d61342` |
| Python | `3.11.9` |
| NumPy | `2.4.6` |
| scikit-learn | `1.9.0` |
| pytest | `9.1.1` |
| ruff | `0.15.20` |

## Dataset Artifacts

These files are intentionally ignored by Git and must not be committed.

| Artifact | Local path | Size | SHA256 |
| --- | --- | ---: | --- |
| Raw Daphnet ZIP | `data/raw/daphnet_fog.zip` | ~20 MB | `27c6173b3acee75a69284903756b9011a00766c1bbb743c37fc884c5dbc39399` |
| Processed trunk CSV | `data/processed/daphnet_trunk.csv` | ~21 MB | `120855a3da4077b01ad663b9e2c924e4ea5e9ffd6610953fb681790367f0a632` |

The processed CSV has `1,140,836` lines including the header, corresponding to
`1,140,835` samples.

## Ignored Result Artifacts

These JSON files are local reproducibility outputs and are intentionally ignored
by Git. Their aggregate values are copied into the committed reports.

| Artifact | Local path | Size | SHA256 |
| --- | --- | ---: | --- |
| 3-fold grouped benchmark JSON | `results/daphnet_trunk_benchmark.json` | ~17 KB | `ee0680369a2a1e47f9e454f684573c465048a02c877d3795585a219ea8a08dcb` |
| LOSO benchmark JSON | `results/daphnet_trunk_loso_benchmark.json` | ~34 KB | `58c3929100559182af177cbfea71da13111023375a3d034ab08071b3739f69cd` |
| Leaky diagnostic JSON | `results/daphnet_trunk_leaky_benchmark.json` | ~1.1 KB | `b7b0a2930aa377b67e9a399fc41035915dc05b00f5f8fbacc446beb9f4a57fe7` |
| Final 1,000-permutation JSON | `results/daphnet_trunk_permutation.json` | ~56 KB | `252ec7cd8cd908c73d4b964633e4d92d04de0252053eb62cbb6870cd15724da9` |

## Committed Aggregate-Only Figures

The SVG files under `docs/figures/` are committed because they contain only
aggregate metrics and no raw or participant-level sensor records.

| Figure | Source |
| --- | --- |
| `docs/figures/daphnet_trunk_benchmark.svg` | `results/daphnet_trunk_benchmark.json` |
| `docs/figures/daphnet_loso_benchmark.svg` | `results/daphnet_trunk_loso_benchmark.json` |
| `docs/figures/daphnet_permutation_null.svg` | `results/daphnet_trunk_permutation.json` |

## Final Permutation Status

The committed permutation report documents the final local run:
`1,000` within-subject permutations with `50` random-forest estimators.

Final aggregate results:

| Model | Observed AUROC | Null mean | Null std | n_perms | p-value |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7221 | 0.5215 | 0.0122 | 1,000 | 0.000999 |
| Random forest | 0.7845 | 0.4471 | 0.0153 | 1,000 | 0.000999 |

The p-value equals `1 / 1001`, the minimum non-zero value available under the
conservative Phipson & Smyth estimator for 1,000 permutations. The corresponding
JSON remains ignored by Git; only aggregate values and the aggregate-only SVG are
committed.
