"""Unit tests for rna_portrait.io.

These cover the input-parsing guarantees stated in the README: the reader
accepts comma-, tab- and whitespace-delimited files, skips a header row, and
normalises gene symbols to upper case. The functions under test depend only on
numpy and pandas, so they run without the pretrained model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from rna_portrait.io import expression_vector, read_gene_expression_table


@pytest.mark.parametrize(
    "content",
    [
        "gene\texpression\nTP53\t1.5\nEGFR\t2.0\n",       # tab-delimited
        "gene,expression\nTP53,1.5\nEGFR,2.0\n",           # comma-delimited
        "gene expression\nTP53 1.5\nEGFR 2.0\n",           # whitespace-delimited
    ],
)
def test_read_gene_expression_table_accepts_delimiters(tmp_path, content):
    path = tmp_path / "expr.txt"
    path.write_text(content)

    values = read_gene_expression_table(path)

    assert values == {"TP53": 1.5, "EGFR": 2.0}


def test_read_gene_expression_table_uppercases_genes(tmp_path):
    path = tmp_path / "expr.tsv"
    path.write_text("tp53\t1.5\n")

    assert read_gene_expression_table(path) == {"TP53": 1.5}


def test_read_gene_expression_table_skips_non_numeric_values(tmp_path):
    path = tmp_path / "expr.tsv"
    path.write_text("gene\texpression\nTP53\thigh\nEGFR\t2.0\n")

    assert read_gene_expression_table(path) == {"EGFR": 2.0}


def test_expression_vector_aligns_and_counts_matches():
    vector, matched = expression_vector(["TP53", "EGFR", "MYC"], {"tp53": 1.5, "myc": 3.0})

    assert matched == 2
    np.testing.assert_allclose(vector, np.array([1.5, 0.0, 3.0], dtype=np.float32))


def test_expression_vector_log1p_clips_negative_values():
    vector, matched = expression_vector(["TP53"], {"TP53": -5.0}, log1p=True)

    assert matched == 1
    # max(value, 0.0) -> 0.0, and log1p(0.0) == 0.0
    np.testing.assert_allclose(vector, np.array([0.0], dtype=np.float32))
