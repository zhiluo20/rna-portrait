#!/usr/bin/env python3
"""Export disease-card sample-level transition outputs.

This analysis reconstructs the sample-level "no disease card" raw disease
semantics from the frozen details table. The column `semantic_disease_family`
matches the no-disease-card resolved counts, while
`semantic_resolved_disease_family` and `semantic_disease_semantic_status`
represent the frozen disease-card output.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


from runtime_paths import (
    ARTIFACT_DIR,
    ARTIFACT_ROOT,
    BACKBONE_DIR,
    BUNDLE_ROOT,
    CODE_DIR,
    R_LIB,
    RUN_CWD,
    SCRIPT_DIR,
    SUPP_DIR,
    VALIDATION_DIR,
    WORKSPACE_ROOT,
)
OUTDIR = SUPP_DIR / "sample_level_disease_transition"

POOLS = {
    "External-180": ARTIFACT_ROOT / "external_180_trimmed_benchmark" / "details.csv",
    "MultiSource-450": ARTIFACT_ROOT / "multisource_450_trimmed_benchmark" / "details.csv",
}
NO_DISEASE_CARD_SUMMARY = ARTIFACT_ROOT / "no_disease_card_ablation" / "summary.json"


def load_no_disease_card_summary() -> Dict[str, dict]:
    data = json.loads(NO_DISEASE_CARD_SUMMARY.read_text(encoding="utf-8"))["results"]
    return {row["pool"]: row for row in data}


def status_order() -> List[str]:
    return ["stable", "mixed", "unsupported"]


def make_transition_tables(details: pd.DataFrame, pool: str) -> Dict[str, pd.DataFrame]:
    base = details.copy()
    base.insert(0, "pool", pool)
    base["raw_disease_family"] = base["semantic_disease_family"].fillna("missing")
    base["resolved_disease_family"] = base["semantic_resolved_disease_family"].fillna("missing")
    base["resolved_status"] = base["semantic_disease_semantic_status"].fillna("missing")

    sample_cols = [
        "pool",
        "file",
        "project",
        "expected_site_family",
        "expected_disease_family",
        "raw_disease_family",
        "resolved_disease_family",
        "resolved_status",
        "semantic_state_family",
        "semantic_state_subprofile",
        "anchor_context_state",
        "site_agree_top1",
        "tumor_agree_top1",
        "disease_agree_top1",
        "semantic_disease_what_to_trust",
        "semantic_disease_biological_readout",
    ]
    sample = base[sample_cols].copy()

    raw_to_status = (
        base.groupby(["raw_disease_family", "resolved_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "raw_disease_family", "resolved_status"], ascending=[False, True, True])
    )
    raw_to_resolved = (
        base.groupby(["raw_disease_family", "resolved_disease_family", "resolved_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "raw_disease_family", "resolved_disease_family"], ascending=[False, True, True])
    )
    state_to_status = (
        base.groupby(["semantic_state_family", "resolved_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["semantic_state_family", "n"], ascending=[True, False])
    )
    anchor_to_status = (
        base.groupby(["anchor_context_state", "resolved_status"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["anchor_context_state", "n"], ascending=[True, False])
    )
    return {
        "sample": sample,
        "raw_to_status": raw_to_status,
        "raw_to_resolved": raw_to_resolved,
        "state_to_status": state_to_status,
        "anchor_to_status": anchor_to_status,
    }


def scale(values: Iterable[int], width: float) -> List[float]:
    vals = list(values)
    total = sum(vals) or 1
    return [v / total * width for v in vals]


def write_status_bar_svg(summary: Dict[str, dict]) -> None:
    width = 980
    height = 310
    left = 210
    bar_w = 620
    bar_h = 30
    gap = 38
    colors = {
        "stable": "#2E7D32",
        "mixed": "#D9822B",
        "unsupported": "#6B7280",
    }
    rows: List[Tuple[str, str, Dict[str, int]]] = []
    for pool, row in summary.items():
        rows.append((pool, "No disease card", row["nodisease_status_counts"]))
        rows.append((pool, "Frozen disease card", row["frozen_status_counts"]))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Disease semantic status before and after disease-card resolution</text>',
    ]
    y = 68
    for pool, label, counts in rows:
        parts.append(
            f'<text x="24" y="{y + 21}" font-family="Arial" font-size="13" fill="#111">{html.escape(pool)} / {html.escape(label)}</text>'
        )
        x = left
        for status in status_order():
            n = int(counts.get(status, 0))
            w = n / max(1, sum(counts.values())) * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colors[status]}"/>')
            if w > 38:
                parts.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 20}" font-family="Arial" font-size="12" text-anchor="middle" fill="#fff">{n}</text>'
                )
            x += w
        parts.append(f'<text x="{left + bar_w + 12}" y="{y + 21}" font-family="Arial" font-size="12" fill="#555">n={sum(counts.values())}</text>')
        y += gap

    legend_y = height - 34
    x = left
    for status in status_order():
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[status]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-family="Arial" font-size="12" fill="#333">{status}</text>')
        x += 125
    parts.append("</svg>")
    (OUTDIR / "status_before_after.svg").write_text("\n".join(parts), encoding="utf-8")


def write_raw_to_status_svg(raw_to_status: pd.DataFrame, pool: str, top_k: int = 8) -> None:
    sub = raw_to_status.copy()
    top_raw = sub.groupby("raw_disease_family")["n"].sum().sort_values(ascending=False).head(top_k).index
    sub.loc[~sub["raw_disease_family"].isin(top_raw), "raw_disease_family"] = "other_raw_families"
    pivot = (
        sub.groupby(["raw_disease_family", "resolved_status"])["n"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=status_order(), fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    width = 1020
    row_h = 32
    top = 70
    left = 280
    bar_w = 560
    height = top + row_h * len(pivot) + 70
    colors = {
        "stable": "#2E7D32",
        "mixed": "#D9822B",
        "unsupported": "#6B7280",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">{html.escape(pool)} raw disease semantics resolved by disease card</text>',
        '<text x="24" y="56" font-family="Arial" font-size="12" fill="#666">Rows show no-disease-card raw disease families; colors show frozen resolved disease status.</text>',
    ]
    y = top
    for raw, row in pivot.iterrows():
        counts = {status: int(row.get(status, 0)) for status in status_order()}
        total = sum(counts.values())
        parts.append(f'<text x="24" y="{y + 20}" font-family="Arial" font-size="12" fill="#111">{html.escape(str(raw))}</text>')
        x = left
        for status in status_order():
            n = counts[status]
            w = n / max(1, total) * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" fill="{colors[status]}"/>')
            if w > 34:
                parts.append(
                    f'<text x="{x + w / 2:.1f}" y="{y + 15}" font-family="Arial" font-size="11" text-anchor="middle" fill="#fff">{n}</text>'
                )
            x += w
        parts.append(f'<text x="{left + bar_w + 12}" y="{y + 16}" font-family="Arial" font-size="11" fill="#555">n={total}</text>')
        y += row_h
    legend_y = height - 24
    x = left
    for status in status_order():
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[status]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-family="Arial" font-size="12" fill="#333">{status}</text>')
        x += 125
    parts.append("</svg>")
    (OUTDIR / f"{pool.lower().replace('-', '_')}_raw_to_status.svg").write_text("\n".join(parts), encoding="utf-8")


def write_summary(all_tables: Dict[str, Dict[str, pd.DataFrame]], no_disease_card_summary: Dict[str, dict]) -> None:
    lines = [
        "# Sample-level disease transition",
        "",
        "## Purpose",
        "",
        "This analysis converts the no-disease-card ablation from summary counts into sample-level transition outputs.",
        "",
        "Key reconstruction check: `semantic_disease_family` in frozen details exactly matches the no-disease-card raw disease-family counts for both External-180 and MultiSource-450.",
        "",
        "## Main findings",
        "",
    ]
    for pool, tables in all_tables.items():
        sample = tables["sample"]
        status_counts = sample["resolved_status"].value_counts().to_dict()
        raw_top = sample["raw_disease_family"].value_counts().head(5).to_dict()
        resolved_top = sample["resolved_disease_family"].value_counts().head(5).to_dict()
        row = no_disease_card_summary[pool]
        lines.extend(
            [
                f"### {pool}",
                "",
                f"- n_samples: `{len(sample)}`",
                f"- no-disease-card raw top disease families: `{raw_top}`",
                f"- frozen resolved top disease families: `{resolved_top}`",
                f"- frozen resolved status counts: `{status_counts}`",
                f"- no-disease-card status counts: `{row['nodisease_status_counts']}`",
                f"- frozen status counts: `{row['frozen_status_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Output files",
            "",
            "- `sample_level_transitions.csv`: sample-level raw-to-resolved transition table.",
            "- `raw_to_resolved_status.csv`: raw disease family by frozen resolved status.",
            "- `raw_to_resolved_family.csv`: raw disease family by frozen resolved disease family and status.",
            "- `state_family_to_status.csv`: semantic state family by frozen resolved status.",
            "- `anchor_context_to_status.csv`: anchor context by frozen resolved status.",
            "- `status_before_after.svg`: summary status before/after disease-card resolution.",
            "- `external_180_raw_to_status.svg` and `multisource_450_raw_to_status.svg`: raw disease families resolved into stable/mixed/unsupported status.",
            "",
            "## Interpretation",
            "",
            "This result supports the claim that raw disease-like semantics are frequently not stable disease identity. The disease card converts raw disease families into overlap, mixed, or unsupported interpretations while preserving broad site/tumor agreement.",
            "",
            "The downstream marker, pathway and cell-composition analyses provide independent biological checks for the mixed and overlap states.",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    no_disease_card_summary = load_no_disease_card_summary()
    all_tables: Dict[str, Dict[str, pd.DataFrame]] = {}
    sample_tables = []
    raw_status_tables = []
    raw_family_tables = []
    state_tables = []
    anchor_tables = []
    for pool, path in POOLS.items():
        details = pd.read_csv(path)
        tables = make_transition_tables(details, pool)
        all_tables[pool] = tables
        sample_tables.append(tables["sample"])
        raw_status_tables.append(tables["raw_to_status"].assign(pool=pool))
        raw_family_tables.append(tables["raw_to_resolved"].assign(pool=pool))
        state_tables.append(tables["state_to_status"].assign(pool=pool))
        anchor_tables.append(tables["anchor_to_status"].assign(pool=pool))

        observed = details["semantic_disease_family"].value_counts().to_dict()
        expected = no_disease_card_summary[pool]["nodisease_resolved_counts"]
        if observed != expected:
            raise RuntimeError(f"{pool}: semantic_disease_family does not match no-disease-card counts")

        write_raw_to_status_svg(tables["raw_to_status"], pool)

    pd.concat(sample_tables, ignore_index=True).to_csv(OUTDIR / "sample_level_transitions.csv", index=False)
    pd.concat(raw_status_tables, ignore_index=True).to_csv(OUTDIR / "raw_to_resolved_status.csv", index=False)
    pd.concat(raw_family_tables, ignore_index=True).to_csv(OUTDIR / "raw_to_resolved_family.csv", index=False)
    pd.concat(state_tables, ignore_index=True).to_csv(OUTDIR / "state_family_to_status.csv", index=False)
    pd.concat(anchor_tables, ignore_index=True).to_csv(OUTDIR / "anchor_context_to_status.csv", index=False)
    write_status_bar_svg(no_disease_card_summary)
    write_summary(all_tables, no_disease_card_summary)


if __name__ == "__main__":
    main()
