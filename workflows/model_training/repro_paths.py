from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "workflows"
BME_CODE_DIR = Path(os.getenv("RNA_PORTRAIT_MODEL_TRAINING_CODE", str(CODE_ROOT / "model_training"))).resolve()
PREPROCESS_CODE_DIR = Path(os.getenv("RNA_PORTRAIT_PREPROCESS_CODE", str(CODE_ROOT / "preprocess"))).resolve()

DATA_ROOT = Path(os.getenv("RNA_PORTRAIT_DATA_ROOT", os.getenv("SRP_DATA_ROOT", str(ROOT / "data")))).resolve()
TRAINING_DATASET_DIR = Path(os.getenv("RNA_PORTRAIT_TRAINING_MATRIX", os.getenv("BMM_DATASET_DIR", str(DATA_ROOT / "training_matrix")))).resolve()
GEO_PROCESSED_SOURCE_DIR = Path(os.getenv("RNA_PORTRAIT_GEO_SOURCE_DIR", os.getenv("SRP_GEO_SOURCE_DIR", str(DATA_ROOT / "geo_processed_source")))).resolve()
TCGA_PROCESSED_SOURCE_DIR = Path(os.getenv("RNA_PORTRAIT_TCGA_SOURCE_DIR", os.getenv("SRP_TCGA_SOURCE_DIR", str(DATA_ROOT / "tcga_processed_source")))).resolve()
TRAINING_RAW_COUNTS_DIR = Path(os.getenv("RNA_PORTRAIT_RAW_COUNTS_DIR", os.getenv("SRP_RAW_COUNTS_DIR", str(DATA_ROOT / "training_raw_counts_with_soft")))).resolve()
EXTERNAL_PROCESSED_DIR = Path(os.getenv("RNA_PORTRAIT_EXTERNAL_PROCESSED_DIR", os.getenv("SRP_EXTERNAL_PROCESSED_DIR", str(DATA_ROOT / "external_processed_multisource")))).resolve()

ARTIFACT_ROOT = Path(os.getenv("RNA_PORTRAIT_ARTIFACT_ROOT", os.getenv("SRP_ARTIFACT_ROOT", str(ROOT / "artifacts")))).resolve()
OUTPUT_ROOT = ARTIFACT_ROOT / "bulk_multimodal_embedding"

VALIDATION_INPUT_ROOT = Path(os.getenv("RNA_PORTRAIT_VALIDATION_INPUT_ROOT", os.getenv("SRP_VALIDATION_INPUT_ROOT", str(ROOT / "validation_inputs")))).resolve()
