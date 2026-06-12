from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = REPO_ROOT / "workflows" / "manuscript_analyses"
FIGURE_SCRIPT = REPO_ROOT / "workflows" / "figures" / "make_main_figures_and_source_data.py"


ANALYSIS_STEPS = [
    ("RNA-language alignment", "01_rna_language_alignment.py"),
    ("Predefined-label baseline", "02_predefined_label_baseline.py"),
    ("Predefined-label calibration", "03_predefined_label_calibration.py"),
    ("Portrait transition summaries", "04_portrait_transition.py"),
    ("Marker-program grounding", "05_marker_program_grounding.py"),
    ("Pathway grounding", "06_pathway_grounding.py"),
    ("Cell-composition signatures", "07_cell_composition_signatures.py"),
    ("EPIC deconvolution", "08_epic_deconvolution.py"),
    ("MCP-counter deconvolution", "09_mcpcounter_deconvolution.py"),
    ("Initial profile-mixing controls", "10_profile_mixing_initial.py"),
    ("Calibrated profile-mixing controls", "11_profile_mixing_calibrated.py"),
    ("Bootstrapped profile-mixing controls", "12_profile_mixing_bootstrap.py"),
    ("Representative case audit", "13_case_audit.py"),
    ("Portrait-claim grounding", "14_portrait_claim_grounding.py"),
    ("Source and metadata controls", "15_source_metadata_controls.py"),
    ("Residual source controls", "16_residual_source_controls.py"),
    ("Public-data quality robustness", "17_quality_robustness.py"),
    ("Reliability boundaries", "18_reliability_boundaries.py"),
    ("Whole-profile versus local features", "19_whole_profile_vs_local_features.py"),
    ("Disease-associated portrait compositions", "20_portrait_composition_disease_information.py"),
]


R_DECONVOLUTION_STEPS = {"08_epic_deconvolution.py", "09_mcpcounter_deconvolution.py"}

R_DEPENDENT_STEPS = {
    "13_case_audit.py",
    "14_portrait_claim_grounding.py",
    "15_source_metadata_controls.py",
    "16_residual_source_controls.py",
    "17_quality_robustness.py",
    "18_reliability_boundaries.py",
    "19_whole_profile_vs_local_features.py",
    "20_portrait_composition_disease_information.py",
}


def run_python(script: Path, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(script)], check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manuscript analyses and regenerate figure source data.")
    parser.add_argument("--data-root", type=Path, default=None, help="Directory containing the training matrix and validation profiles.")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts", help="Directory containing or receiving trained model artifacts.")
    parser.add_argument("--analysis-output-root", type=Path, default=REPO_ROOT / "outputs" / "manuscript_analysis_tables")
    parser.add_argument("--figure-output-root", type=Path, default=REPO_ROOT / "outputs" / "figures")
    parser.add_argument("--r-lib-root", type=Path, default=None, help="Optional R library root containing EPIC and MCP-counter dependencies.")
    parser.add_argument("--skip-r-deconvolution", action="store_true", help="Skip EPIC and MCP-counter analyses when R dependencies are unavailable.")
    parser.add_argument("--skip-figures", action="store_true", help="Run analyses only.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["RNA_PORTRAIT_REPO_ROOT"] = str(REPO_ROOT)
    env["RNA_PORTRAIT_ARTIFACT_ROOT"] = str(args.artifact_root.resolve())
    env["RNA_PORTRAIT_ANALYSIS_OUTPUT_ROOT"] = str(args.analysis_output_root.resolve())
    env["RNA_PORTRAIT_RESULTS_ROOT"] = str(args.analysis_output_root.resolve())
    env["RNA_PORTRAIT_FIGURE_OUTPUT_ROOT"] = str(args.figure_output_root.resolve())
    env["RNA_PORTRAIT_MODEL_TRAINING_CODE"] = str((REPO_ROOT / "workflows" / "model_training").resolve())
    default_alignment_dir = args.artifact_root.resolve() / "bulk_multimodal_embedding" / "semantic_alignment_backbone"
    if default_alignment_dir.exists():
        env.setdefault("RNA_PORTRAIT_ALIGNMENT_RUN_DIR", str(default_alignment_dir))
    if args.r_lib_root is not None:
        env["RNA_PORTRAIT_R_LIB_ROOT"] = str(args.r_lib_root.resolve())
    if args.data_root is not None:
        env["RNA_PORTRAIT_DATA_ROOT"] = str(args.data_root.resolve())
        env["RNA_PORTRAIT_TRAINING_MATRIX"] = str((args.data_root / "training_matrix").resolve())
        env["RNA_PORTRAIT_VALIDATION_INPUT_ROOT"] = str((args.data_root / "validation_profiles").resolve())

    for label, filename in ANALYSIS_STEPS:
        if args.skip_r_deconvolution and filename in R_DECONVOLUTION_STEPS:
            print(f"[skip] {label} (R deconvolution disabled)", flush=True)
            continue
        if args.skip_r_deconvolution and filename in R_DEPENDENT_STEPS:
            print(f"[skip] {label} (requires EPIC/MCP-counter outputs)", flush=True)
            continue
        print(f"[run] {label}", flush=True)
        run_python(ANALYSIS_DIR / filename, env)

    if not args.skip_figures:
        if args.skip_r_deconvolution:
            print("[skip] Main figure source-data and figure generation (requires EPIC/MCP-counter outputs)", flush=True)
        else:
            print("[run] Main figure source-data and figure generation", flush=True)
            run_python(FIGURE_SCRIPT, env)


if __name__ == "__main__":
    main()
