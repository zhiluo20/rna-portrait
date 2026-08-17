from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path


EXPECTED_PDFS = (
    "Figure_1_architecture.pdf",
    "Figure_2_RNA_language_alignment.pdf",
    "Figure_3_biological_grounding.pdf",
    "Figure_4_portraits_not_single_labels.pdf",
    "Figure_5_stress_tests_and_reliability.pdf",
    "Extended_Data_Fig_1.pdf",
    "Extended_Data_Fig_2.pdf",
    "Extended_Data_Fig_3.pdf",
    "Extended_Data_Fig_4.pdf",
    "Extended_Data_Fig_5.pdf",
    "Extended_Data_Fig_6.pdf",
    "Extended_Data_Fig_7.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_pdf_checks(path: Path) -> list[str]:
    errors: list[str] = []
    if pdfinfo := shutil.which("pdfinfo"):
        result = subprocess.run(
            [pdfinfo, str(path)], capture_output=True, check=False, text=True
        )
        match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
        if result.returncode or match is None:
            errors.append(f"{path.name}: pdfinfo could not read the file")
        elif int(match.group(1)) != 1:
            errors.append(f"{path.name}: expected one page, found {match.group(1)}")

    if pdffonts := shutil.which("pdffonts"):
        result = subprocess.run(
            [pdffonts, str(path)], capture_output=True, check=False, text=True
        )
        if result.returncode:
            errors.append(f"{path.name}: pdffonts could not read the file")
        else:
            rows = result.stdout.splitlines()[2:]
            embedded = []
            for row in rows:
                match = re.search(
                    r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row
                )
                if match:
                    embedded.append(match.group(1) == "yes")
            if not embedded:
                errors.append(f"{path.name}: no font records found")
            elif not all(embedded):
                errors.append(f"{path.name}: contains an unembedded font")
    return errors


def validate_figure_directory(root: Path, *, run_external: bool = True) -> list[str]:
    errors: list[str] = []
    if not root.is_dir():
        return [f"not a directory: {root}"]

    expected = set(EXPECTED_PDFS)
    observed = {path.name for path in root.glob("*.pdf") if not path.name.startswith("._")}
    for name in sorted(expected - observed):
        errors.append(f"missing PDF: {name}")
    for name in sorted(observed - expected):
        errors.append(f"unexpected PDF: {name}")

    for name in EXPECTED_PDFS:
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size < 8 or not path.read_bytes()[:5] == b"%PDF-":
            errors.append(f"{name}: not a readable PDF header")
            continue
        if run_external:
            errors.extend(_external_pdf_checks(path))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the 12-PDF figure set used by the current Word manuscript."
    )
    parser.add_argument("figure_dir", type=Path)
    args = parser.parse_args()

    errors = validate_figure_directory(args.figure_dir.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    for name in EXPECTED_PDFS:
        path = args.figure_dir.resolve() / name
        print(f"{sha256(path)}  {name}")
    print(f"PASS: {len(EXPECTED_PDFS)} manuscript PDFs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
