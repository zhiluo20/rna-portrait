from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def release_files():
    for path in [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CODE_AVAILABILITY_STATEMENT.txt",
        REPO_ROOT / "DATA_AVAILABILITY_STATEMENT.txt",
        REPO_ROOT / "CITATION.cff",
        REPO_ROOT / "MODEL_LICENSE",
    ]:
        yield path
    yield from (
        path
        for path in (REPO_ROOT / "workflows").rglob("*.py")
        if not path.name.startswith("._")
    )
    yield from (
        path
        for path in (REPO_ROOT / "workflows").rglob("*.md")
        if not path.name.startswith("._")
    )


def test_documented_release_entrypoints_exist():
    expected = [
        "workflows/preprocessing/prepare_geo_matrix.py",
        "workflows/preprocessing/process_tcga_gdc.py",
        "workflows/preprocessing/build_merged_training_dataset.py",
        "workflows/preprocessing/materialize_merged_expr_matrix.py",
        "workflows/run_training_pipeline.py",
        "workflows/model_training/run_runtime_validation_pipeline.py",
        "workflows/run_manuscript_pipeline.py",
        "workflows/figures/current_submission/make_submission_figures.py",
        "notebooks/External_Independent_Reproduction_Guide.ipynb",
    ]
    assert [path for path in expected if not (REPO_ROOT / path).is_file()] == []


def test_release_text_has_no_author_workstation_paths_or_mutable_downloads():
    forbidden = [
        "/Volumes/",
        "/Users/",
        "/opt/ryzen",
        "MCPcounter/master",
    ]
    hits = []
    for path in release_files():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(REPO_ROOT)}: {token}")
    assert hits == []


def test_release_has_direct_figure_dependency_and_pinned_text_encoder():
    requirements = (REPO_ROOT / "requirements.txt").read_text()
    training = (
        REPO_ROOT
        / "workflows/model_training/train_rna_language_alignment.py"
    ).read_text()
    assert "Pillow>=10" in requirements
    assert "sentence-transformers>=3.0.1,<4" in requirements
    assert "TEXT_MODEL_REVISION" in training
    assert "c9745ed1d9f207416be6d2e6f8de32d1f16199bf" in training


def test_public_guide_contains_no_saved_execution_outputs():
    notebook = json.loads(
        (REPO_ROOT / "notebooks/External_Independent_Reproduction_Guide.ipynb").read_text()
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)


def test_model_weight_license_is_explicit():
    text = (REPO_ROOT / "MODEL_LICENSE").read_text()
    assert "Suggested license" not in text
    assert "CC BY-NC 4.0" in text
