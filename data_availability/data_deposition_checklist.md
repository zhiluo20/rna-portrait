# Data and Artifact Deposition Checklist

This checklist separates the clean GitHub code archive from the larger data and
runtime artifacts required for complete manuscript reproduction.

## Deposit in a data or model-artifact repository

- `processed_data/training_matrix/`
  - `expr_log.parquet`
  - `meta.csv`
  - `meta.parquet`
  - `genes.npy`
  - `sample_ids.npy`
  - `summary.json`
- `processed_data/validation_profiles/external_180/`
  - `README.md`
  - `manifest.json`
  - `*.txt`
  - `*.meta.json`
- `processed_data/validation_profiles/multisource_450/`
  - `README.md`
  - `manifest.json`
  - `*.txt`
  - `*.meta.json`
- `model_artifacts/bulk_multimodal_embedding/`
  - `semantic_alignment_backbone/`
  - `semantic_backbone_v8_topk64/`
  - `semantic_state_evidence_scorer_max/`
  - `train_age_adapter_curated_ageq_batch18_sbert_allminilm_v2/`
  - `external_180_trimmed_benchmark/`
  - `multisource_450_trimmed_benchmark/`
  - `no_disease_card_ablation/`
  - `semantic_mainline_best_20260417.json`
  - `semantic_state_evidence_priors_20260418.json`
- `source_data/`
  - all CSV files used to draw main and extended figures
  - `source_data_manifest.csv`
- `manuscript_analysis_tables/`
  - regenerated CSV, TSV, JSON and Markdown summary outputs used to build the
    manuscript figures and tables
- `software_environment/r_libraries_deconvolution/` or equivalent lock files
  - required to reproduce the EPIC and MCP-counter analyses exactly
- `software_environment/third_party_models/`
  - a text file recording the exact upstream revision of
    `sentence-transformers/all-MiniLM-L6-v2`, or a repository-acceptable model
    snapshot manifest if offline reproduction is required

## Keep in the GitHub code archive

- `rna_portrait/`
- `workflows/`
- `models/rna_language_alignment/`
- `models/portrait_attention/`
- `environment.yml`
- `requirements.txt`
- `pyproject.toml`
- `README.md`
- `CODE_AVAILABILITY_STATEMENT.txt`
- `DATA_AVAILABILITY_STATEMENT.txt`
- `data_availability/public_dataset_manifest.csv`
- `data_availability/reproduction_asset_manifest.csv`

## Exclude from deposition unless the editor asks for them

- auxiliary working documents
- exploratory reports not listed in `reproduction_asset_manifest.csv`
- local notebook scratch outputs
- Python `__pycache__/` folders
- `.DS_Store` and macOS `._*` AppleDouble files
- temporary independent-test outputs that are not listed in
  `reproduction_asset_manifest.csv`

## Fields to finalize before submission

- Replace `[GitHub URL]` in the code statement.
- Replace `[CODE ARCHIVE DOI]` after creating the permanent code archive.
- Replace `[DATA REPOSITORY NAME]` and `[DATA REPOSITORY DOI]` after depositing
  processed data, source data and runtime artifacts.
- Replace `[MODEL/DATA REPOSITORY DOI]` if model artifacts are deposited as a
  separate record.
- Record the exact upstream revision or snapshot for
  `sentence-transformers/all-MiniLM-L6-v2`.
