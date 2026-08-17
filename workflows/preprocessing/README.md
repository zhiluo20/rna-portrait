# Training-matrix preprocessing

These scripts provide the deterministic path from public-source expression
exports to the prepared matrix consumed by `workflows/run_training_pipeline.py`.
No generative model or generated annotation is required. Descriptive sample
text is assembled from released metadata fields.

## Inputs

- GEO: a study-level expression matrix with genes on rows and samples on
  columns, plus a metadata table containing `sample_id` and `gse`. The
  expression transform must be established during source-specific processing
  and recorded in the data-package metadata.
- TCGA: GDC STAR gene-count exports with their GDC metadata, clinical and
  biospecimen JSON files. The directory layout accepted by
  `process_tcga_gdc.py` is `normal/` and `tumor/<group>/`, each with `meta/`
  and count files.
- Shared feature space: the ordered `genes.npy` produced for the GEO matrix.

Primary sequence files remain in their original public repositories. The
repository manifest maps the study/project identifiers used in training to
those public sources.

## Run

Harmonize the GEO matrix:

```bash
python workflows/preprocessing/prepare_geo_matrix.py \
  --expression /path/to/geo_expression.parquet \
  --metadata /path/to/geo_metadata.csv \
  --outdir work/geo_processed_source
```

Process GDC STAR unstranded counts as log1p(CPM), using the same ordered gene
space:

```bash
python workflows/preprocessing/process_tcga_gdc.py \
  --tcga-root /path/to/gdc_exports \
  --genes work/geo_processed_source/genes.npy \
  --outdir work/tcga_processed_source
```

Merge the two sources into project shards and then materialize the monolithic
matrix expected by the default training command:

```bash
python workflows/preprocessing/build_merged_training_dataset.py \
  --geo-root work/geo_processed_source \
  --tcga-root work/tcga_processed_source \
  --out-root work/training_matrix

python workflows/preprocessing/materialize_merged_expr_matrix.py \
  --data-dir work/training_matrix
```

The final directory contains:

- `expr_log.parquet`: genes by samples;
- `meta.csv` and `meta.parquet`: aligned sample metadata;
- `genes.npy` and `sample_ids.npy`: ordered identifiers;
- `projects/<project>/shard_<number>/`: sharded copies used for lower-memory
  training;
- `summary.json`: dimensions and layout.

The deposited processed matrix is the Methods-level starting point for a model
rerun. Reconstructing every GEO study from its primary repository additionally
requires the accession-specific normalization choices recorded with the
released data.
