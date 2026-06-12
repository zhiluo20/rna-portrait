from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def read_gene_expression_table(path: str | Path) -> dict[str, float]:
    """Read a two-column gene-expression file into a gene-to-value dictionary.

    The reader accepts comma-, tab- or whitespace-delimited files. The first two
    columns are interpreted as gene symbol and numeric expression value.
    """

    path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            table = pd.read_csv(path, sep=None, engine="python", header=None, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "could not decode input file")

    if table.shape[1] < 2:
        raise ValueError("Expected at least two columns: gene and expression value.")

    values: dict[str, float] = {}
    for _, row in table.iloc[:, :2].iterrows():
        gene = str(row.iloc[0]).strip()
        if not gene or gene.lower() in {"gene", "genes", "symbol"}:
            continue
        try:
            values[gene.upper()] = float(row.iloc[1])
        except Exception:
            continue
    return values


def expression_vector(
    selected_genes: list[str],
    expression_values: Mapping[str, float],
    log1p: bool = False,
) -> tuple[np.ndarray, int]:
    """Align a gene-expression dictionary to the genes expected by the model."""

    upper_values = {str(g).upper(): float(v) for g, v in expression_values.items()}
    vector = np.zeros((len(selected_genes),), dtype=np.float32)
    matched = 0
    for index, gene in enumerate(selected_genes):
        value = upper_values.get(str(gene).upper())
        if value is None:
            continue
        vector[index] = np.log1p(max(value, 0.0)) if log1p else value
        matched += 1
    return vector, matched
