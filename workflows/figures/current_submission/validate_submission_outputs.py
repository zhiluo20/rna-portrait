#!/usr/bin/env python3
"""Validate the exact current 5+7 figure set and its source-data manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


FIGURES = [
    "Figure_1_architecture",
    "Figure_2_RNA_language_alignment",
    "Figure_3_biological_grounding",
    "Figure_4_portraits_not_single_labels",
    "Figure_5_stress_tests_and_reliability",
    *[f"Extended_Data_Fig_{number}" for number in range(1, 8)],
]
FORMATS = ["svg", "pdf", "png", "tiff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    return parser.parse_args()


def validate_manifest(source_data: Path) -> None:
    manifest = source_data / "source_data_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    listed = [Path(row["draft_copy"]).name for row in rows]
    actual = sorted(
        path.name for path in source_data.glob("*.csv")
        if path.name != manifest.name and not path.name.startswith("._")
    )
    if len(listed) != len(set(listed)):
        raise SystemExit("Duplicate entries in source_data_manifest.csv")
    if sorted(listed) != actual:
        raise SystemExit(
            "Source-data manifest mismatch: "
            f"unlisted={sorted(set(actual) - set(listed))}; missing={sorted(set(listed) - set(actual))}"
        )


def validate_figures(figure_dir: Path) -> None:
    expected = {f"{name}.{extension}" for name in FIGURES for extension in FORMATS}
    actual = {
        path.name for path in figure_dir.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower().lstrip(".") in FORMATS
    }
    if actual != expected:
        raise SystemExit(
            f"Figure set mismatch: missing={sorted(expected - actual)}; unexpected={sorted(actual - expected)}"
        )
    for name in FIGURES:
        with Image.open(figure_dir / f"{name}.tiff") as image:
            if image.mode != "RGB":
                raise SystemExit(f"{name}.tiff is {image.mode}, expected RGB")
            if image.info.get("compression") != "tiff_lzw":
                raise SystemExit(f"{name}.tiff is not LZW-compressed")


def main() -> None:
    args = parse_args()
    validate_manifest(args.source_data.resolve())
    validate_figures(args.figure_dir.resolve())
    print("validated: 5 main + 7 Extended Data figures in SVG/PDF/PNG/TIFF; source-data manifest complete")


if __name__ == "__main__":
    main()
