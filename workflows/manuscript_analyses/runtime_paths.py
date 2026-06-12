"""Runtime paths for the manuscript reproduction scripts.

Workflow reruns can redirect new artifacts and figure tables to an isolated
runtime output directory by setting the environment variables below.
"""

from __future__ import annotations

import os
from pathlib import Path


BUNDLE_ROOT = Path(os.getenv("RNA_PORTRAIT_REPO_ROOT", os.getenv("SRP_BUNDLE_ROOT", str(Path(__file__).resolve().parents[2])))).resolve()
WORKSPACE_ROOT = BUNDLE_ROOT
SCRIPT_DIR = Path(__file__).resolve().parent

CODE_DIR = Path(os.getenv("RNA_PORTRAIT_MODEL_TRAINING_CODE", str(WORKSPACE_ROOT / "workflows" / "model_training"))).resolve()

ARTIFACT_BASE = Path(os.getenv("RNA_PORTRAIT_ARTIFACT_ROOT", os.getenv("SRP_ARTIFACT_ROOT", str(WORKSPACE_ROOT / "artifacts")))).resolve()
ARTIFACT_DIR = ARTIFACT_BASE / "bulk_multimodal_embedding"
ARTIFACT_ROOT = ARTIFACT_DIR


def default_alignment_run_dir() -> Path:
    for name in ["semantic_alignment_backbone", "rna_language_alignment"]:
        candidate = ARTIFACT_DIR / name
        if candidate.exists():
            return candidate
    return ARTIFACT_DIR / "semantic_alignment_backbone"


BACKBONE_DIR = Path(os.getenv("RNA_PORTRAIT_ALIGNMENT_RUN_DIR", str(default_alignment_run_dir()))).resolve()

VALIDATION_DIR = Path(os.getenv("RNA_PORTRAIT_VALIDATION_INPUT_ROOT", os.getenv("SRP_VALIDATION_INPUT_ROOT", str(WORKSPACE_ROOT / "validation_inputs")))).resolve()
SUPP_DIR = Path(os.getenv("RNA_PORTRAIT_ANALYSIS_OUTPUT_ROOT", os.getenv("SRP_ANALYSIS_RESULTS_DIR", str(WORKSPACE_ROOT / "outputs" / "manuscript_analysis_tables")))).resolve()
R_LIB = Path(os.getenv("RNA_PORTRAIT_R_LIB_ROOT", os.getenv("SRP_R_LIB_ROOT", str(WORKSPACE_ROOT / "envs" / "r_libs_deconv")))).resolve()
RUN_CWD = Path(os.getenv("RNA_PORTRAIT_RUNTIME_CWD", os.getenv("SRP_RUNTIME_CWD", str(WORKSPACE_ROOT)))).resolve()
