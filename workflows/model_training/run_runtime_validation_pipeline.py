#!/usr/bin/env python3
"""Build semantic benchmark, evidence-support and ablation artifacts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def run(script: str, arguments: list[str], env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script), *arguments],
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        type=Path,
        required=True,
        help="Directory containing external_180/ and multisource_450/.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Build benchmark/scorer/prior artifacts without the no-disease-card ablation.",
    )
    args = parser.parse_args()

    validation_root = args.validation_root.resolve()
    artifact_root = args.artifact_root.resolve()
    output_root = artifact_root / "bulk_multimodal_embedding"
    for pool in ["external_180", "multisource_450"]:
        if not (validation_root / pool / "manifest.json").is_file():
            parser.error(f"Missing validation manifest: {validation_root / pool / 'manifest.json'}")

    env = os.environ.copy()
    env["RNA_PORTRAIT_ARTIFACT_ROOT"] = str(artifact_root)
    env["RNA_PORTRAIT_VALIDATION_INPUT_ROOT"] = str(validation_root)
    env["RNA_PORTRAIT_MODEL_TRAINING_CODE"] = str(SCRIPT_DIR)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rna_portrait_bootstrap_", dir=output_root) as tmp:
        bootstrap = Path(tmp)
        bootstrap_env = env.copy()
        bootstrap_env["SEMANTIC_DISABLE_EVIDENCE_SCORER"] = "1"
        bootstrap_env["SEMANTIC_DISABLE_EVIDENCE_PRIORS"] = "1"
        run(
            "benchmark_semantic_unknown_explainer_expanded.py",
            [
                "--validation-dir",
                str(validation_root / "multisource_450"),
                "--outdir",
                str(bootstrap),
            ],
            bootstrap_env,
        )
        run(
            "train_semantic_state_evidence_scorer.py",
            [
                "--details",
                str(bootstrap / "details.csv"),
                "--outdir",
                str(output_root / "semantic_state_evidence_scorer_max"),
            ],
            env,
        )
        run(
            "build_semantic_state_evidence_priors.py",
            [
                "--details",
                str(bootstrap / "details.csv"),
                "--out",
                str(output_root / "semantic_state_evidence_priors_20260418.json"),
            ],
            env,
        )

    for pool, output_name in [
        ("external_180", "external_180_trimmed_benchmark"),
        ("multisource_450", "multisource_450_trimmed_benchmark"),
    ]:
        run(
            "benchmark_semantic_unknown_explainer_expanded.py",
            [
                "--validation-dir",
                str(validation_root / pool),
                "--outdir",
                str(output_root / output_name),
            ],
            env,
        )

    if not args.skip_ablation:
        run("run_ablation_no_disease_card.py", [], env)


if __name__ == "__main__":
    main()
