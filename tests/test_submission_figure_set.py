from workflows.figures.check_submission_figure_set import (
    EXPECTED_PDFS,
    validate_figure_directory,
)


def write_minimal_pdf_set(root):
    for name in EXPECTED_PDFS:
        (root / name).write_bytes(b"%PDF-1.4\n")


def test_submission_figure_inventory_accepts_exact_set(tmp_path):
    write_minimal_pdf_set(tmp_path)
    assert validate_figure_directory(tmp_path, run_external=False) == []


def test_submission_figure_inventory_reports_missing_and_extra(tmp_path):
    write_minimal_pdf_set(tmp_path)
    (tmp_path / EXPECTED_PDFS[0]).unlink()
    (tmp_path / "old_figure.pdf").write_bytes(b"%PDF-1.4\n")
    errors = validate_figure_directory(tmp_path, run_external=False)
    assert errors == [
        f"missing PDF: {EXPECTED_PDFS[0]}",
        "unexpected PDF: old_figure.pdf",
    ]
