# Historical normalization recovery

The deposited training matrix is not the output of the published GDC `log1p(CPM)` route. Retained source files and count matrices recovered on 13 October 2026 establish a project/cohort-wise log-median-ratio normalization followed by `log1p`. The code and checkpoints in the preceding release remain unchanged.

This note concerns reconstruction of historical training inputs. It does not report a new training run, establish a prospective single-sample input contract, or validate model accuracy. Previously reported reference-CPM pilot results must not be relabeled as training-matched results without rerunning an explicitly defined evaluation.

## What was recovered

| Evidence | Historical behavior | Difference from the published route |
| --- | --- | --- |
| Original `normalize_tcga_deseq2style.py` and saved factors | Calls `ov.bulk.estimateSizeFactors` and `ov.bulk.deseq2_normalize` on each complete GDC project, then stores `log1p` values as float32 | `process_tcga_gdc.py` computes CPM instead |
| Original `build_merged_training_dataset.py` | Copies `expr_log_deseq2_omicverse.parquet` into the merged `expr_log.parquet` layout | The published merge script selects `expr_log1p_cpm.parquet` |
| GEO gene-symbol count CSVs, normalized CSVs and prepared matrix | All 48 retained GEO cohorts numerically match the same estimator, with float64 log-expression | `prepare_geo_matrix.py` preserves a supplied transformation; it does not reconstruct the upstream normalization |

The recovered OmicVerse source checkout identifies itself as `1.7.10rc1`. That source snapshot and its license have been preserved locally. This does not prove which complete environment was installed when the historical matrices were produced. The original GEO execution driver and its environment lock have not been recovered; numerical agreement is not presented as recovery of those missing records.

## Numerical checks

All 69 GDC-derived projects were recomputed from 88 recovered count shards. Their 20,356 samples and 30,286 genes account for 616,501,816 values. Using the two inspected OmicVerse functions from the preserved source snapshot, the reconstructed float32 values equal every deposited GDC value. The largest difference between recomputed and saved size factors is approximately 9.77e-15.

All 48 GEO cohorts were checked from their gene-symbol count tables. Reconstructed normalized counts match the saved normalized CSVs within the recorded floating-point tolerance. Their `log1p` values match the prepared GEO matrix with a maximum absolute difference of approximately 3.55e-15. The prepared matrix and all deposited GEO shards match exactly across 146,675,098 values from 4,843 samples. Gene and sample identities were aligned explicitly; row position was not used as a join key.

Aggregate evidence and the independent implementation validation are recorded in [the machine-readable audit summary](../data_availability/normalization_recovery_20261013.json). These numerical checks do not verify every original download, raw identifier mapping, sample exclusion or historical software dependency.

The independent implementation included in this change was additionally tested on all 117 cohorts and 25,199 samples. Across 763,176,914 values, GDC float32 values match exactly and GEO values match exactly after conversion to the model's float32 input precision. The largest GEO float64 difference for this independent array implementation is 1.95e-14. This is separately recorded from the 3.55e-15 result obtained with the inspected OmicVerse functions.

## Reconstruct a complete cohort

[reconstruct_legacy_normalization.py](../workflows/preprocessing/reconstruct_legacy_normalization.py) independently implements the recovered mathematical transformation with NumPy and pandas. It does not import or vendor OmicVerse. Existing repository dependencies are sufficient.

For genes positive in every supplied sample, it computes the mean log count across the complete cohort. Each sample's size factor is the exponential of its median log ratio to those means. All count rows are divided by the resulting sample factor and transformed with natural-log `log1p`.

Choose the historical precision explicitly: float32 for the GDC-derived component and float64 for the prepared GEO component. Preserve the original gene/sample identifiers and the complete gene universe used to fit that cohort. Do not fit independently on the 512-sample storage shards. A storage shard is not a normalization cohort.

The following small example is a synthetic software fixture, not study data:

```python
import pandas as pd
from workflows.preprocessing.reconstruct_legacy_normalization import normalize_cohort

counts = pd.DataFrame({"S1": [1, 4, 0], "S2": [4, 16, 2]}, index=["A", "B", "C"])
logged, factors, reference = normalize_cohort(counts, output_dtype="float64")
print(factors.to_list())  # approximately [0.5, 2.0]
```

The CLI accepts a complete gene-by-sample cohort count matrix in CSV or Parquet. Run `python workflows/preprocessing/reconstruct_legacy_normalization.py --help` for its required arguments. Its new output directory contains `expr_log.parquet`, `size_factors.parquet`, `reference_mean_log_counts.parquet` and `normalization_manifest.json`. The manifest records the input hash, implementation hash, software versions, cohort dimensions, precision, formula and output hashes. Existing output directories are rejected rather than overwritten.

Run the software checks with:

```bash
python -m pytest tests/test_legacy_normalization.py -q
```

Inputs with duplicate identifiers, missing/negative/nonfinite values, fractional counts, or no gene positive throughout the cohort fail explicitly. Single-sample input reproduces the estimator's factor of one and is flagged in the manifest; it is not evidence that this is a useful normalization rule for prospective single-sample inference.

## Remaining work and boundaries

- The utility is a cohort-normalization component, not a replacement for raw-source ingestion, GEO selection, complete matrix assembly or model training.
- Historical factors were fitted within complete source cohorts. A training-only reference, deployment-time behavior, controlled gene-loss handling and train/validation/test separation must be explicitly defined before a new preprocessing/model version is evaluated.
- Upstream identifier mapping, sample selection, the exact historical package environment and the anchor backbone's original split provenance are not closed by this audit.
- Independent human assessment and confirmatory retrieval utility remain incomplete.
- Retained participant-level data, count tables, model weights and vendor source snapshots are not uploaded in this change. Source hashes and bounded aggregate validation are published; the full local audit preserves the input files and detailed records.

The manuscript, model, data and code versions remain separate. A future preprocessing or training change must receive its own identifier and record its input hashes, split manifests, parameters, seeds, environment, logs, checkpoint hashes and corresponding Git commit. It must not overwrite the submitted model or reuse the old benchmark results as though they were newly measured.
