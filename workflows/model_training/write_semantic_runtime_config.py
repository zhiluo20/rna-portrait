#!/usr/bin/env python3
"""Write the semantic runtime configuration used by the explainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Root containing bulk_multimodal_embedding/.",
    )
    parser.add_argument("--name", default="semantic_mainline_best_20260417")
    parser.add_argument("--attention-run", default="semantic_backbone_v8_topk64")
    parser.add_argument("--base-text-source", default="metadata_text")
    parser.add_argument("--fusion-alpha", type=float, default=0.4)
    args = parser.parse_args()
    if not 0.0 <= args.fusion_alpha <= 1.0:
        parser.error("--fusion-alpha must be between 0 and 1")

    output_root = args.artifact_root.resolve() / "bulk_multimodal_embedding"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "semantic_mainline_best_20260417.json"
    payload = {
        "name": args.name,
        "base_text_source": args.base_text_source,
        "attention_run": args.attention_run,
        "fusion_alpha": args.fusion_alpha,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
