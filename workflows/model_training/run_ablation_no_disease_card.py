#!/usr/bin/env python3
"""Compare semantic runtime outputs with and without the disease card."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from repro_paths import OUTPUT_ROOT as OUT
from repro_paths import VALIDATION_INPUT_ROOT


CODE = Path(__file__).resolve().parent
PYTHON = sys.executable
BENCH = CODE / "benchmark_semantic_unknown_explainer_expanded.py"

POOLS = {
    "External-180": VALIDATION_INPUT_ROOT / "external_180",
    "MultiSource-450": VALIDATION_INPUT_ROOT / "multisource_450",
}


def run_one(pool_name: str, validation_dir: Path, disable_disease_card: bool, tag: str) -> Path:
    outdir = OUT / f"ablation_no_disease_card_{pool_name.lower().replace('-', '_')}_{tag}_20260419"
    env = os.environ.copy()
    env["SEMANTIC_DISABLE_DISEASE_CARD"] = "1" if disable_disease_card else "0"
    subprocess.run(
        [
            PYTHON,
            str(BENCH),
            "--validation-dir",
            str(validation_dir),
            "--outdir",
            str(outdir),
        ],
        check=True,
        env=env,
    )
    return outdir


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def main() -> None:
    rows = []
    for pool_name, validation_dir in POOLS.items():
        out_ablation = run_one(pool_name, validation_dir, True, "nodiseasecard")
        out_frozen = run_one(pool_name, validation_dir, False, "frozen")
        s0 = load_summary(out_ablation)
        s1 = load_summary(out_frozen)
        rows.append(
            {
                "pool": pool_name,
                "n_samples": s1["n_samples"],
                "n_projects": s1["n_projects"],
                "nodisease_site": s0["site_agree_top1_rate"],
                "nodisease_tumor": s0["tumor_agree_top1_rate"],
                "nodisease_disease": s0["disease_agree_top1_rate"],
                "nodisease_resolved_counts": s0["semantic_resolved_disease_family_counts"],
                "nodisease_status_counts": s0["semantic_disease_semantic_status_counts"],
                "nodisease_subprofile_counts": s0["semantic_disease_semantic_subprofile_counts"],
                "frozen_site": s1["site_agree_top1_rate"],
                "frozen_tumor": s1["tumor_agree_top1_rate"],
                "frozen_disease": s1["disease_agree_top1_rate"],
                "frozen_resolved_counts": s1["semantic_resolved_disease_family_counts"],
                "frozen_status_counts": s1["semantic_disease_semantic_status_counts"],
                "frozen_subprofile_counts": s1["semantic_disease_semantic_subprofile_counts"],
            }
        )

    outdir = OUT / "no_disease_card_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps({"results": rows}, indent=2, ensure_ascii=False))
    lines = ["# No-disease-card ablation", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['pool']}",
                "",
                f"- samples/projects: `{row['n_samples']} / {row['n_projects']}`",
                f"- no-disease-card site/tumor/disease: `{row['nodisease_site']:.4f} / {row['nodisease_tumor']:.4f} / {row['nodisease_disease']:.4f}`",
                f"- frozen site/tumor/disease: `{row['frozen_site']:.4f} / {row['frozen_tumor']:.4f} / {row['frozen_disease']:.4f}`",
                f"- no-disease-card status counts: `{row['nodisease_status_counts']}`",
                f"- frozen status counts: `{row['frozen_status_counts']}`",
                "",
            ]
        )
    (outdir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
