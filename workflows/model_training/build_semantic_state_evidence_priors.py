#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from repro_paths import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETAILS = OUTPUT_ROOT / "multisource_450_trimmed_benchmark" / "details.csv"
DEFAULT_OUT = OUTPUT_ROOT / "semantic_state_evidence_priors_20260418.json"
EVIDENCE_KINDS = [
    "semantic_majority",
    "anchor_context",
    "neighbor_context",
    "age_driver_overlap",
    "tension",
    "anchor_override",
]


def _smoothed_dist(series: pd.Series, alpha: float = 1.0) -> dict[str, float]:
    counts = series.astype(str).value_counts().to_dict()
    total = float(sum(counts.values()) + alpha * len(EVIDENCE_KINDS))
    return {
        kind: float((counts.get(kind, 0.0) + alpha) / total)
        for kind in EVIDENCE_KINDS
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.details)
    if df.empty:
        raise SystemExit("empty details.csv")

    family_priors = {
        str(family): _smoothed_dist(group["semantic_state_top_evidence_kind"])
        for family, group in df.groupby("semantic_state_family")
    }
    subprofile_priors = {
        str(subprofile): _smoothed_dist(group["semantic_state_top_evidence_kind"])
        for subprofile, group in df.groupby("semantic_state_subprofile")
    }

    payload = {
        "source_details": args.details.name,
        "n_samples": int(len(df)),
        "evidence_kinds": EVIDENCE_KINDS,
        "family_priors": family_priors,
        "subprofile_priors": subprofile_priors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
