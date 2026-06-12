# RNA Portrait Code Availability

This repository contains the code and pretrained model files used for the manuscript
**Gene expression profiles can be described in natural language like images**.

The repository is organized for code availability. Processed data, validation
profiles, full manuscript runtime artifacts and figure source data required for
complete reproduction should be deposited separately under the manuscript Data
Availability statement.

## Repository contents

- `rna_portrait/`: importable Python package for loading the pretrained RNA-language
  portrait model and describing new bulk RNA profiles.
- `models/`: pretrained RNA-language alignment and portrait-attention weights used
  for inference.
- `workflows/model_training/`: training code for the RNA-language alignment backbone,
  portrait-attention module and optional age-like readout.
- `workflows/manuscript_analyses/`: manuscript analysis scripts ordered by the
  analysis sequence.
- `workflows/figures/`: scripts that regenerate main-figure source data and figures
  from manuscript analysis tables.
- `notebooks/External_Independent_Reproduction_Guide.ipynb`: step-by-step
  notebook for external verification from the released code and separate data
  package.
- `data_availability/public_dataset_manifest.csv`: public dataset/source manifest
  derived from the training metadata.
- `data_availability/reproduction_asset_manifest.csv`: exact processed-data,
  model-artifact, source-data and environment assets required for complete
  manuscript reproduction.
- `DATA_AVAILABILITY_STATEMENT.txt`: Data Availability statement text with
  placeholders for the final repository DOI/accession.
- `examples/example_gene_expression.tsv`: minimal two-column input example.

## Install

```bash
conda env create -f environment.yml
conda activate rna-portrait
pip install -e .
```

or:

```bash
pip install -e .
```

## Run pretrained inference

```python
import rna_portrait as rp

model = rp.load_model("models")
result = model.describe_file("examples/example_gene_expression.tsv", top_k=5)
print(result["portrait_text"])
print(result["portrait_summary"])
```

Input files should contain at least two columns: gene symbol and expression value.
Comma-, tab- and whitespace-delimited files are accepted.

## Train the model from a prepared training matrix

The training matrix directory should contain the files used by the Methods:

- `expr_log.parquet`
- `meta.csv`
- `genes.npy`
- `sample_ids.npy`

Run:

```bash
python workflows/run_training_pipeline.py \
  --training-matrix /path/to/training_matrix \
  --artifact-root artifacts
```

The pipeline trains:

1. RNA-language alignment backbone.
2. Portrait-attention model.
3. Optional age-like readout.

## External reproduction notebook

For independent verification, use:

```text
notebooks/External_Independent_Reproduction_Guide.ipynb
```

Place the data package next to this repository:

```text
parent_directory/
  nature_code_availability_20260611/
  nature_reproduction_data_package_20260611/
```

Then open the notebook and run cells in order. The notebook:

1. loads and previews the prepared bulk RNA training matrix;
2. checks the public-data and reproduction-asset manifests;
3. runs the full-data training workflow when `RUN_MODEL_TRAINING = True`;
4. can rerun the manuscript analysis workflow when
   `RUN_MANUSCRIPT_PIPELINE = True`;
5. displays standard matplotlib result panels directly from CSV source data.

If your directory names differ, set the environment variables
`RNA_PORTRAIT_CODE_REPO`, `RNA_PORTRAIT_DATA_PACKAGE` and
`RNA_PORTRAIT_WORK_DIR`, or edit the first notebook cell.

## Reproduce manuscript analyses

The analysis pipeline expects trained model artifacts, validation profiles and
prepared data tables described in the Data Availability statement.

```bash
python workflows/run_manuscript_pipeline.py \
  --data-root /path/to/released_processed_data \
  --artifact-root /path/to/model_artifacts \
  --r-lib-root /path/to/released_r_libraries_deconvolution \
  --analysis-output-root outputs/manuscript_analysis_tables \
  --figure-output-root outputs/figures
```

If R packages for EPIC or MCP-counter are unavailable, run:

```bash
python workflows/run_manuscript_pipeline.py \
  --data-root /path/to/released_processed_data \
  --artifact-root /path/to/model_artifacts \
  --skip-r-deconvolution
```

## Manuscript analysis order

The scripts in `workflows/manuscript_analyses/` follow the manuscript analysis sequence:

1. RNA-language alignment.
2. Predefined disease-label baseline and calibration.
3. Molecular portrait transitions.
4. Marker, pathway and deconvolution grounding.
5. Controlled profile mixing.
6. Representative case audit.
7. Source, tissue, platform and metadata controls.
8. Robustness and reliability boundaries.
9. Whole-profile versus local-feature comparisons.
10. Disease-associated portrait compositions.

## Data and artifact availability

Complete manuscript reproduction requires assets that are intentionally not
stored directly in this clean GitHub code archive. Deposit these assets in a
data or model-artifact repository and keep the release paths consistent with
`data_availability/reproduction_asset_manifest.csv`:

- prepared training matrix: `processed_data/training_matrix/`
- external validation profiles: `processed_data/validation_profiles/`
- full manuscript runtime artifacts: `model_artifacts/bulk_multimodal_embedding/`
- figure source data: `source_data/`
- regenerated manuscript analysis tables: `manuscript_analysis_tables/`
- R deconvolution environment or lock files:
  `software_environment/r_libraries_deconvolution/`
- third-party text-encoder revision record:
  `software_environment/third_party_models/`

The compact `models/` directory in this code archive is sufficient for example
inference. It is not, by itself, sufficient to reproduce every manuscript
analysis.

## Licenses

Source code is released under the MIT License. Model weights are governed by
`MODEL_LICENSE`; users should ensure that reuse complies with the terms of the
underlying public datasets and any institutional requirements.

## Permanent archive

For submission, create a GitHub release and archive that release on Zenodo or
Code Ocean. Nature Portfolio guidance states that a GitHub URL alone is not a
permanent identifier; the archived DOI should be cited in the manuscript.
