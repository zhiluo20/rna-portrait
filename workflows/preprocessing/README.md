# Training-matrix preprocessing

## Historical reconstruction status

The deposited GDC-derived training expression uses whole-project OmicVerse-style size-factor normalization followed by `log1p`, not the CPM output produced by `process_tcga_gdc.py` below. Retained counts and normalization records recovered on 13 October 2026 support this distinction. The original merge script selected `expr_log_deseq2_omicverse.parquet`; the published merge script selects the CPM file.

See [the recovery note](../../docs/preprocessing_recovery_20261013.md) and [aggregate verification](../../data_availability/normalization_recovery_20261013.json) before attempting a historical rebuild. [reconstruct_legacy_normalization.py](reconstruct_legacy_normalization.py) provides an independently tested cohort-normalization component with explicit output precision and checksum-bound run records. It does not replace upstream source processing or establish a prospective single-sample input contract.

The commands below are retained to document the published CPM/harmonization route. They must not be described as reconstructing the deposited training matrix. No pretrained checkpoint or existing transformation has been silently replaced.

## Inputs to the published alternative route

- GEO: a study-level expression matrix with genes on rows and samples on columns, plus a metadata table containing `sample_id` and `gse`. The expression transform must be established during source-specific processing and recorded in the data-package metadata.
- TCGA/GDC: STAR gene-count exports with their metadata, clinical and biospecimen JSON files. The directory layout accepted by `process_tcga_gdc.py` is `normal/` and `tumor/<group>/`, each with `meta/` and count files.
- Shared feature space: the ordered `genes.npy` produced for the GEO matrix.

Primary sequence files remain in their original public repositories. The repository manifest maps the study/project identifiers used in training to those public sources.

## Published alternative commands

Harmonize a supplied, already transformed GEO matrix:

```bash
python workflows/preprocessing/prepare_geo_matrix.py --help
```

This script standardizes symbols, averages duplicates and stores float32 values. It is not the recovered upstream GEO normalization driver and should not be used to claim byte-identical reconstruction of the original float64 GEO matrix.

Inspect the GDC CPM processor and the matrix assembly commands:

```bash
python workflows/preprocessing/process_tcga_gdc.py --help
python workflows/preprocessing/build_merged_training_dataset.py --help
python workflows/preprocessing/materialize_merged_expr_matrix.py --help
```

The merge script still selects `expr_log1p_cpm.parquet`. Do not relabel this file as size-factor-normalized expression. Preserve outputs in a separate versioned location; the existing assembly scripts can replace project directories in their output root.

## Training input layout

The training command consumes:

- `expr_log.parquet`: genes by samples;
- `meta.csv` and `meta.parquet`: aligned sample metadata;
- `genes.npy` and `sample_ids.npy`: ordered identifiers;
- `projects/<project>/shard_<number>/`: sharded copies;
- `summary.json`: dimensions and layout.

The deposited prepared matrix remains the starting point for reproducing the frozen historical model. Restoring its normalization provenance does not resolve all original sample-selection, identifier-mapping or environment records. New models and evaluations require separate versioned manifests and cannot inherit the old results without rerunning them.
