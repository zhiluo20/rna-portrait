from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workflows.preprocessing.prepare_geo_matrix import (
    align_metadata,
    harmonize_expression,
)
from workflows.preprocessing.process_tcga_gdc import counts_to_log1p_cpm


def test_harmonize_expression_uppercases_and_averages_duplicate_genes():
    expr = pd.DataFrame(
        {"S1": [1.0, 3.0, 5.0], "S2": [2.0, 4.0, 6.0]},
        index=[" tp53 ", "TP53", "egfr"],
    )

    out = harmonize_expression(expr)

    assert list(out.index) == ["EGFR", "TP53"]
    np.testing.assert_allclose(out.loc["TP53"], [2.0, 3.0])
    assert out.dtypes.tolist() == [np.dtype("float32"), np.dtype("float32")]


def test_harmonize_expression_rejects_non_numeric_values():
    expr = pd.DataFrame({"S1": ["not-a-number"]}, index=["TP53"])

    with pytest.raises(ValueError, match="missing or non-numeric"):
        harmonize_expression(expr)


def test_align_metadata_requires_exact_sample_set_and_order():
    expr = pd.DataFrame({"S2": [1.0], "S1": [2.0]}, index=["TP53"])
    meta = pd.DataFrame(
        {"sample_id": ["S1", "S2"], "gse": ["GSE1", "GSE1"]}
    )

    out = align_metadata(expr, meta)

    assert out["sample_id"].tolist() == ["S2", "S1"]


def test_align_metadata_rejects_missing_samples():
    expr = pd.DataFrame({"S1": [1.0]}, index=["TP53"])
    meta = pd.DataFrame({"sample_id": ["S2"], "gse": ["GSE1"]})

    with pytest.raises(ValueError, match="sample mismatch"):
        align_metadata(expr, meta)


def test_counts_to_log1p_cpm_normalizes_each_sample():
    counts = pd.DataFrame(
        {"S1": [1.0, 3.0], "S2": [0.0, 2.0]},
        index=["TP53", "EGFR"],
    )

    out = counts_to_log1p_cpm(counts)

    expected = np.log1p(
        np.array([[250_000.0, 0.0], [750_000.0, 1_000_000.0]])
    )
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)
