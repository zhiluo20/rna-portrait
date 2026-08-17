#!/usr/bin/env python3
"""Build the five main and seven Extended Data figures in the current Word manuscript."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STEPS = [
    "figure_1.py",
    "main_figures.py",
    "figure_3.py",
    "extended_data_other.py",
    "extended_data_4.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-package", type=Path, required=True,
                        help="Released data-package root containing source_data and figure_panel_sources")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Clean output directory for SVG, PDF, PNG and TIFF figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = args.data_package.resolve()
    output = args.output_dir.resolve()
    source_data = package / "source_data"
    panel_sources = package / "figure_panel_sources"
    if not source_data.is_dir():
        raise SystemExit(f"Missing source-data directory: {source_data}")
    if not panel_sources.is_dir():
        raise SystemExit(f"Missing vector-panel source directory: {panel_sources}")
    output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["RNA_PORTRAIT_SOURCE_DATA_DIR"] = str(source_data)
    env["RNA_PORTRAIT_PANEL_SOURCE_DIR"] = str(panel_sources)
    env["RNA_PORTRAIT_FIGURE_OUTPUT_ROOT"] = str(output)
    for step in STEPS:
        print(f"[figure workflow] {step}", flush=True)
        subprocess.run([sys.executable, str(HERE / step)], check=True, env=env)

    subprocess.run(
        [sys.executable, str(HERE / "validate_submission_outputs.py"),
         "--figure-dir", str(output), "--source-data", str(source_data)],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
