from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = REPO_ROOT / "workflows" / "model_training"


def run(script: str, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, str(TRAINING_DIR / script)], check=True, cwd=str(REPO_ROOT), env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNA-language alignment and portrait-attention models.")
    parser.add_argument("--training-matrix", type=Path, required=True, help="Directory containing expr_log.parquet, meta.csv, genes.npy and sample_ids.npy.")
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--alignment-run-name", default="rna_language_alignment")
    parser.add_argument("--portrait-run-name", default="portrait_attention")
    parser.add_argument("--skip-age-readout", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["RNA_PORTRAIT_MODEL_TRAINING_CODE"] = str(TRAINING_DIR.resolve())
    env["RNA_PORTRAIT_ARTIFACT_ROOT"] = str(args.artifact_root.resolve())
    env["RNA_PORTRAIT_TRAINING_MATRIX"] = str(args.training_matrix.resolve())
    env["BMM_DATASET_DIR"] = str(args.training_matrix.resolve())
    env["BMM_RUN_NAME"] = args.alignment_run_name
    env["SPTA_BASE_RUN"] = args.alignment_run_name
    env["SPTA_RUN_NAME"] = args.portrait_run_name
    env.setdefault("SPTA_TEXT_SOURCE", "metadata_text")
    env.setdefault("SPTA_N_PROTOTYPES", "256")
    env.setdefault("SPTA_TOPK_PROTOTYPES", "64")

    print("[run] RNA-language alignment backbone")
    run("train_rna_language_alignment.py", env)
    print("[run] Portrait-attention model")
    run("train_portrait_attention.py", env)
    if not args.skip_age_readout:
        print("[run] Optional age-like readout")
        run("train_age_readout.py", env)


if __name__ == "__main__":
    main()
