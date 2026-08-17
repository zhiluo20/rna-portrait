# RNA Portrait Code Availability

This repository contains the code and pretrained model files used for the manuscript
**Can genes be described in natural language like images?**

The repository is organized for code availability. Processed data, validation
profiles, full runtime artifacts and figure source data are deposited in a
private Zenodo record with reserved DOI
[10.5281/zenodo.20695991](https://doi.org/10.5281/zenodo.20695991). Controlled
access is provided during peer review, with public release planned on
publication. Verify the deposited contents against
`data_availability/reproduction_asset_manifest.csv`.

## Repository contents

- `rna_portrait/`: importable Python package for loading the pretrained RNA-language
  portrait model and describing new bulk RNA profiles.
- `models/`: pretrained RNA-language alignment and portrait-attention weights used
  for inference.
- `workflows/model_training/`: training code for the RNA-language alignment backbone,
  portrait-attention module, semantic runtime support artifacts and optional
  age-like readout.
- `workflows/preprocessing/`: deterministic GEO matrix harmonization, GDC
  STAR-count processing and GEO/TCGA training-matrix assembly.
- `workflows/manuscript_analyses/`: manuscript analysis scripts ordered by the
  analysis sequence.
- `workflows/figures/current_submission/`: Python/Matplotlib reconstruction of
  the five main figures and seven Extended Data figures from released inputs.
- `notebooks/External_Independent_Reproduction_Guide.ipynb`: step-by-step
  integrity check, figure map and 5+7 figure rebuild.
- `data_availability/public_dataset_manifest.csv`: public dataset/source manifest
  derived from the training metadata.
- `data_availability/reproduction_asset_manifest.csv`: exact processed-data,
  model-artifact, source-data and environment assets required for complete
  manuscript reproduction.
- `CODE_AVAILABILITY_STATEMENT.txt` and
  `DATA_AVAILABILITY_STATEMENT.txt`: manuscript-ready availability text.
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

It also writes the semantic runtime configuration used by the explainer.
Build the validation benchmarks, evidence-support scorer, evidence priors and
no-disease-card ablation with:

```bash
python workflows/model_training/run_runtime_validation_pipeline.py \
  --validation-root /path/to/released_data_package/processed_data/validation_profiles \
  --artifact-root artifacts
```

To reconstruct the prepared matrix from public-source exports, follow
`workflows/preprocessing/README.md`. The deposited prepared matrix is the
Methods-level starting point for a model rerun; accession-specific GEO
normalization choices remain documented with the deposited data.

## External reproduction notebook

The external reproduction notebook is:

```text
notebooks/External_Independent_Reproduction_Guide.ipynb
```

Place the data package next to this repository:

```text
parent_directory/
  nature_code_availability_20260611/
  released_data_package/
```

Running the notebook:

1. verifies the source-data manifest;
2. checks the prepared training matrix and released vector-panel sources;
3. records the Fig. 1–5 and Extended Data Fig. 1–7 mapping;
4. rebuilds and validates all 12 figures in SVG, PDF, PNG and RGB/LZW TIFF;
5. performs quantitative spot checks and renders a visual inspection sheet.

If your directory names differ, set the environment variables
`RNA_PORTRAIT_CODE_REPO`, `RNA_PORTRAIT_DATA_PACKAGE` and
`RNA_PORTRAIT_NOTEBOOK_OUTPUT`, or edit the first notebook cell.

## Reproduce manuscript analyses

The analysis pipeline expects trained model artifacts, validation profiles and
prepared data tables described in the Data Availability statement.

```bash
python workflows/run_manuscript_pipeline.py \
  --data-root /path/to/released_data_package/processed_data \
  --data-package-root /path/to/released_data_package \
  --artifact-root /path/to/model_artifacts \
  --r-lib-root /path/to/reconstructed_r_library \
  --analysis-output-root outputs/manuscript_analysis_tables \
  --figure-output-root outputs/figures
```

If R packages for EPIC or MCP-counter are unavailable, run:

```bash
python workflows/run_manuscript_pipeline.py \
  --data-root /path/to/released_data_package/processed_data \
  --data-package-root /path/to/released_data_package \
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
stored directly in this clean GitHub code archive. The companion Zenodo record
(reserved DOI `10.5281/zenodo.20695991`) contains the following paths, as
recorded in `data_availability/reproduction_asset_manifest.csv`:

- prepared training matrix: `processed_data/training_matrix/`
- external validation profiles: `processed_data/validation_profiles/`
- full manuscript runtime artifacts: `model_artifacts/bulk_multimodal_embedding/`
- figure source data: `source_data/`
- vector-panel sources used by the Extended Data reconstruction:
  `figure_panel_sources/`
- regenerated manuscript analysis tables: `manuscript_analysis_tables/`
- R deconvolution version record and reconstruction notes:
  `software_environment/r_package_versions.csv` and
  `software_environment/README_r_environment.md`
- third-party text-encoder revision record:
  `software_environment/third_party_models/`

The deposited release archive is
`rna_portrait_data_package_20260817.tar.gz`; its SHA-256 checksum is provided in
`rna_portrait_data_package_20260817.tar.gz.sha256`.

The compact `models/` directory in this code archive is sufficient for example
inference. It is not, by itself, sufficient to reproduce every manuscript
analysis. Exact figure reconstruction uses the matching Zenodo-version package
together with `workflows/figures/current_submission/`.

Validate a flat manuscript PDF folder with:

```bash
python workflows/figures/check_submission_figure_set.py /path/to/figure_pdfs
```

The checker requires the five manuscript main-figure names and seven
Extended Data figure names, verifies one page per PDF when `pdfinfo` is
available, and checks font embedding when `pdffonts` is available.

For permanent citation, archive the reviewed GitHub release and add its software
DOI to `CITATION.cff`. The reserved Zenodo DOI above identifies the data package
and must not be presented as a software-release DOI.

## Licenses

Source code is released under the MIT License. Model weights are governed by
`MODEL_LICENSE`; users should ensure that reuse complies with the terms of the
underlying public datasets and any institutional requirements.
