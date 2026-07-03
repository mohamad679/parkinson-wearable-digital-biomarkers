# Daphnet Artifact Manifest

> **Research-use boundary:** This manifest documents local, ignored artifacts used
> to reproduce the Daphnet trunk-sensor benchmark. It is not clinical evidence and
> does not contain raw sensor rows or participant-level records.

## Repository State

| Item | Value |
| --- | --- |
| Audited branch | `agent/phase-1-2-3-validation-ablation` |
| Audited commit before manifest | `e7686b9c35073082125848718dc2eadd7babf7ab` |
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
| Fast permutation JSON | `results/daphnet_trunk_permutation.json` | ~6.8 KB | `43c33503e0851dec797ffa9bc918005efd8e2df693dd28049e9b6a3803b0e444` |

## Committed Aggregate-Only Figures

The SVG files under `docs/figures/` are committed because they contain only
aggregate metrics and no raw or participant-level sensor records.

| Figure | Source |
| --- | --- |
| `docs/figures/daphnet_trunk_benchmark.svg` | `results/daphnet_trunk_benchmark.json` |
| `docs/figures/daphnet_loso_benchmark.svg` | `results/daphnet_trunk_loso_benchmark.json` |
| `docs/figures/daphnet_permutation_null.svg` | `results/daphnet_trunk_permutation.json` |

## Current Permutation Status

The committed permutation report currently documents the fast CPU run:
`100` permutations with `10` random-forest estimators.

A final local rerun is being performed separately with:

```bash
python scripts/permutation_test.py \
  --input data/processed/daphnet_trunk.csv \
  --output results/daphnet_trunk_permutation_1000.json \
  --sampling-rate 64 \
  --window-size 128 \
  --overlap 0.5 \
  --folds 3 \
  --n-permutations 1000 \
  --random-seed 42 \
  --random-forest-estimators 50

```

After that run completes, update:
- `docs/daphnet_permutation_report.md`
- `docs/figures/daphnet_permutation_null.svg`
- this manifest section

