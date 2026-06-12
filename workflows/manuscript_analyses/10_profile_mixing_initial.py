#!/usr/bin/env python3
"""Controlled in silico mixing analysis for closed-set overcall stress testing."""

from __future__ import annotations

import html
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier


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
OUTDIR = SUPP_DIR / "T5_controlled_mixing"

T2_SCRIPT = SCRIPT_DIR / "02_predefined_label_baseline.py"
T4_SCORES = SUPP_DIR / "T4_marker_program_state_validation" / "t4_marker_scores_by_sample.csv"
EXPLAINER_SCRIPT = CODE_DIR / "infer_unknown_rna_semantic_explainer.py"

RNG_SEED = 20260604
N_REPLICATE_PAIRS = 3
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]

INPUT_DIRS = {
    "External-180": VALIDATION_DIR / "external_180",
    "MultiSource-450": VALIDATION_DIR / "multisource_450",
}


def load_module(path: Path, module_name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def softmax_decision(clf: SGDClassifier, x: np.ndarray) -> Tuple[str, float]:
    scores = clf.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    idx = int(probs.argmax(axis=1)[0])
    return str(clf.classes_[idx]), float(probs[0, idx])


def parse_gene_value_file(path: Path) -> pd.Series:
    rows: List[Tuple[str, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            gene, value = line.split(",", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            gene, value = parts[0], parts[1]
        try:
            rows.append((gene.strip().upper(), float(value)))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no gene-value rows parsed from {path}")
    return pd.DataFrame(rows, columns=["gene", "value"]).groupby("gene")["value"].sum().astype(np.float32)


def transform_external_values(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=np.float32)
    frac_nonint = float(np.mean(np.abs(arr - np.round(arr)) > 1e-6))
    vmax = float(np.max(arr)) if len(arr) else 0.0
    if vmax > 50 or frac_nonint < 0.1:
        return np.log1p(values.clip(lower=0)).astype(np.float32)
    return values.astype(np.float32)


@lru_cache(maxsize=512)
def load_profile(pool: str, file_name: str) -> pd.Series:
    return parse_gene_value_file(INPUT_DIRS[pool] / file_name)


def selected_gene_matrix(values: pd.Series, ckpt: dict) -> np.ndarray:
    genes = [str(g).upper() for g in ckpt["selected_genes"]]
    selected = pd.Series(0.0, index=genes, dtype=np.float32)
    transformed = transform_external_values(values)
    overlap = transformed.index.intersection(selected.index)
    selected.loc[overlap] = transformed.loc[overlap].astype(np.float32)
    mean = np.asarray(ckpt["expr_mean"], dtype=np.float32)
    std = np.asarray(ckpt["expr_std"], dtype=np.float32)
    std = np.where(std < 1e-3, 1.0, std)
    return np.clip((selected.to_numpy(dtype=np.float32) - mean) / std, -8.0, 8.0)[None, :]


def mix_profiles(a: pd.Series, b: pd.Series, fraction_b: float) -> pd.Series:
    idx = a.index.union(b.index)
    av = a.reindex(idx, fill_value=0.0).astype(np.float32)
    bv = b.reindex(idx, fill_value=0.0).astype(np.float32)
    return ((1.0 - fraction_b) * av + fraction_b * bv).astype(np.float32)


def add_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z = lambda c: pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["tumor_program_score"] = z("epithelial_z") + z("stromal_ecm_z") + z("proliferation_z")
    out["immune_program_score"] = z("immune_core_z") + z("t_cell_nk_z") + z("hematologic_lineage_z") + z("myeloid_inflammation_z")
    out["clean_program_score"] = (
        z("epithelial_z").clip(lower=0.0)
        - z("immune_core_z").abs()
        - z("hematologic_lineage_z").abs()
        - z("proliferation_z").abs()
    )
    return out


def top_unique(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    return (
        df.sort_values(score_col, ascending=False)
        .drop_duplicates(["pool", "file"])
        .head(n)
        .reset_index(drop=True)
    )


def select_component_samples(scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = add_selection_scores(scores)
    disease = df["expected_disease_family"].astype(str)
    site = df["expected_site_family"].astype(str)
    state = df["semantic_state_family"].astype(str)
    status = df["semantic_disease_semantic_status"].astype(str)

    tumor = top_unique(
        df.loc[disease.str.contains("tumor|malignancy", case=False, regex=True)],
        "tumor_program_score",
        N_REPLICATE_PAIRS,
    )
    immune = top_unique(
        df.loc[(site == "hematologic") | state.str.contains("hematologic|immune", case=False, regex=True)],
        "immune_program_score",
        N_REPLICATE_PAIRS,
    )
    normal_mask = (disease == "healthy_control") & (status == "stable")
    normal = top_unique(df.loc[normal_mask], "clean_program_score", N_REPLICATE_PAIRS)
    if len(normal) < N_REPLICATE_PAIRS:
        normal = top_unique(df.loc[disease.isin(["healthy_control", "other_non_tumor"])], "clean_program_score", N_REPLICATE_PAIRS)
    if min(len(tumor), len(immune), len(normal)) < N_REPLICATE_PAIRS:
        raise RuntimeError("not enough component samples for controlled mixing")
    return tumor, immune, normal


def component_row(row: pd.Series, role: str, rank: int) -> dict:
    return {
        f"{role}_pool": str(row["pool"]),
        f"{role}_file": str(row["file"]),
        f"{role}_project": str(row["project"]),
        f"{role}_expected_disease_family": str(row["expected_disease_family"]),
        f"{role}_expected_site_family": str(row["expected_site_family"]),
        f"{role}_semantic_state_family": str(row["semantic_state_family"]),
        f"{role}_semantic_disease_status": str(row["semantic_disease_semantic_status"]),
        f"{role}_rank": int(rank),
    }


def build_mixing_design(scores: pd.DataFrame) -> pd.DataFrame:
    tumor, immune, normal = select_component_samples(scores)
    rows = []
    for i in range(N_REPLICATE_PAIRS):
        rank = i + 1
        t = tumor.iloc[i]
        im = immune.iloc[i]
        n = normal.iloc[i]
        rows.append(
            {
                "design": "tumor_normal",
                "fraction_label": "tumor_fraction",
                **component_row(n, "component_a", rank),
                **component_row(t, "component_b", rank),
            }
        )
        rows.append(
            {
                "design": "tumor_immune",
                "fraction_label": "immune_fraction",
                **component_row(t, "component_a", rank),
                **component_row(im, "component_b", rank),
            }
        )
        rows.append(
            {
                "design": "normal_immune",
                "fraction_label": "immune_fraction",
                **component_row(n, "component_a", rank),
                **component_row(im, "component_b", rank),
            }
        )
    return pd.DataFrame(rows)


def train_closed_set_baseline():
    t2 = load_module(T2_SCRIPT, "t5_t2_helpers")
    train_module = t2.load_module(CODE_DIR / "train_rna_language_alignment.py", "t5_predefined_label_train_module")
    taxonomy = t2.load_module(CODE_DIR / "family_taxonomy.py", "t5_closed_set_family_taxonomy")
    ckpt = t2.torch_load(t2.CHECKPOINT)
    x_train, train_meta = t2.prepare_training_matrix(train_module, ckpt, split="train")
    y_train = t2.make_training_labels(train_meta, taxonomy)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=45,
        tol=1e-3,
        class_weight="balanced",
        random_state=RNG_SEED,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    return clf, ckpt


def extract_openworld(payload: dict) -> dict:
    consensus = payload.get("semantic_consensus", {}) or {}
    explanation = payload.get("semantic_explanation", {}) or {}
    state_card = explanation.get("semantic_state_card", {}) or {}
    disease_resolution = explanation.get("disease_semantic_resolution", {}) or {}
    anchor = payload.get("structured_anchor", {}) or {}
    top_evidence = (state_card.get("evidence_highlights", []) or [{}])[0]
    return {
        "openworld_route": str(consensus.get("route", "unknown")),
        "openworld_route_subtype": str(consensus.get("route_subtype", "unknown")),
        "openworld_state_family": str(state_card.get("state_family", "unknown")),
        "openworld_state_subprofile": str(state_card.get("state_subprofile", "unknown")),
        "openworld_resolved_disease_family": str(disease_resolution.get("resolved_disease_family", "unknown")),
        "openworld_disease_status": str(disease_resolution.get("status", "unknown")),
        "openworld_disease_subprofile": str(disease_resolution.get("subprofile", "unknown")),
        "openworld_anchor_context_state": str(anchor.get("context_state", "unknown")),
        "openworld_anchor_tumor_status": str(anchor.get("tumor_status", "unknown")),
        "openworld_state_top_evidence_kind": str(top_evidence.get("kind", "unknown")),
        "openworld_state_top_evidence_score": float(top_evidence.get("score", 0.0) or 0.0),
        "openworld_state_top_evidence_strength": float(top_evidence.get("strength", 0.0) or 0.0),
        "matched_selected_genes": int(payload.get("matched_selected_genes", 0)),
    }


def run_predictions(design: pd.DataFrame, clf: SGDClassifier, ckpt: dict) -> pd.DataFrame:
    explainer = load_module(EXPLAINER_SCRIPT, "t5_semantic_explainer")
    rows = []
    for pair_idx, rec in enumerate(design.itertuples(index=False), start=1):
        a = load_profile(str(rec.component_a_pool), str(rec.component_a_file))
        b = load_profile(str(rec.component_b_pool), str(rec.component_b_file))
        for frac in FRACTIONS:
            sample_id = f"{rec.design}_pair{pair_idx:02d}_f{int(round(frac * 100)):03d}"
            values = mix_profiles(a, b, frac)
            x = selected_gene_matrix(values, ckpt)
            closed_label, closed_conf = softmax_decision(clf, x)
            counts = {str(g): float(v) for g, v in values.items() if np.isfinite(v)}
            payload = explainer.explain_counts(counts, top_k=5, rerank_beta=0.3)
            openworld = extract_openworld(payload)
            row = {
                "synthetic_sample_id": sample_id,
                "design": str(rec.design),
                "pair_index": int(pair_idx),
                "fraction_label": str(rec.fraction_label),
                "fraction_b": float(frac),
                "component_a_file": str(rec.component_a_file),
                "component_b_file": str(rec.component_b_file),
                "component_a_expected_disease_family": str(rec.component_a_expected_disease_family),
                "component_b_expected_disease_family": str(rec.component_b_expected_disease_family),
                "closed_set_disease_family": closed_label,
                "closed_set_confidence": closed_conf,
                "closed_set_forced_stable": True,
                **openworld,
            }
            row["openworld_stable"] = row["openworld_disease_status"] == "stable"
            row["openworld_mixed_or_unsupported"] = row["openworld_disease_status"] != "stable"
            row["closed_set_highconf_overcall_70"] = bool(closed_conf >= 0.70 and row["openworld_mixed_or_unsupported"])
            rows.append(row)
            print(
                f"[profile-mixing] {sample_id}: predefined={closed_label}({closed_conf:.3f}) "
                f"portrait={row['openworld_disease_status']}/{row['openworld_state_family']}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (design, fraction), sub in pred.groupby(["design", "fraction_b"], dropna=False):
        rows.append(
            {
                "design": design,
                "fraction_b": float(fraction),
                "n": int(len(sub)),
                "mean_closed_set_confidence": float(sub["closed_set_confidence"].mean()),
                "closed_set_highconf_overcall_70_rate": float(sub["closed_set_highconf_overcall_70"].mean()),
                "openworld_stable_rate": float(sub["openworld_stable"].mean()),
                "openworld_mixed_or_unsupported_rate": float(sub["openworld_mixed_or_unsupported"].mean()),
                "most_common_closed_set_disease_family": str(sub["closed_set_disease_family"].mode().iloc[0]),
                "most_common_openworld_state_family": str(sub["openworld_state_family"].mode().iloc[0]),
                "most_common_openworld_disease_status": str(sub["openworld_disease_status"].mode().iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["design", "fraction_b"])


def svg_closed_open_curve(summary: pd.DataFrame, path: Path) -> None:
    width, height = 980, 430
    left, top, plot_w, plot_h = 78, 60, 720, 280
    designs = ["tumor_normal", "tumor_immune", "normal_immune"]
    colors = {
        "tumor_normal": "#1565C0",
        "tumor_immune": "#D9822B",
        "normal_immune": "#2E7D32",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Controlled mixing exposes closed-set overcall against open-world disease status</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    def xmap(v: float) -> float:
        return left + v * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - v * plot_h

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{xmap(tick):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.2f}</text>')
        parts.append(f'<line x1="{xmap(tick):.1f}" y1="{top}" x2="{xmap(tick):.1f}" y2="{top + plot_h}" stroke="#eee"/>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{left - 12}" y="{ymap(tick) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{ymap(tick):.1f}" x2="{left + plot_w}" y2="{ymap(tick):.1f}" stroke="#eee"/>')

    for design in designs:
        sub = summary.loc[summary["design"] == design].sort_values("fraction_b")
        if sub.empty:
            continue
        color = colors[design]
        pts_conf = [f"{xmap(float(r.fraction_b)):.1f},{ymap(float(r.mean_closed_set_confidence)):.1f}" for r in sub.itertuples(index=False)]
        pts_over = [f"{xmap(float(r.fraction_b)):.1f},{ymap(float(r.openworld_mixed_or_unsupported_rate)):.1f}" for r in sub.itertuples(index=False)]
        parts.append(f'<path d="M {" L ".join(pts_conf)}" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<path d="M {" L ".join(pts_over)}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="7 5"/>')
        for r in sub.itertuples(index=False):
            parts.append(f'<circle cx="{xmap(float(r.fraction_b)):.1f}" cy="{ymap(float(r.mean_closed_set_confidence)):.1f}" r="4" fill="{color}"/>')
            parts.append(f'<rect x="{xmap(float(r.fraction_b)) - 3:.1f}" y="{ymap(float(r.openworld_mixed_or_unsupported_rate)) - 3:.1f}" width="6" height="6" fill="{color}"/>')
    lx = width - 178
    for i, design in enumerate(designs):
        y = 78 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{colors[design]}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(design)}</text>')
    parts.append(f'<line x1="{lx}" y1="178" x2="{lx + 34}" y2="178" stroke="#555" stroke-width="3"/>')
    parts.append(f'<text x="{lx + 42}" y="182" font-family="Arial" font-size="12">closed-set confidence</text>')
    parts.append(f'<line x1="{lx}" y1="206" x2="{lx + 34}" y2="206" stroke="#555" stroke-width="3" stroke-dasharray="7 5"/>')
    parts.append(f'<text x="{lx + 42}" y="210" font-family="Arial" font-size="12">open mixed/unsupported</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" font-family="Arial" font-size="13" text-anchor="middle">fraction of component B</text>')
    parts.append('<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">rate or score</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_state_trajectory(pred: pd.DataFrame, path: Path) -> None:
    pred = pred.sort_values(["design", "pair_index", "fraction_b"]).copy()
    row_keys = pred[["design", "pair_index"]].drop_duplicates().itertuples(index=False)
    rows = [(str(r.design), int(r.pair_index)) for r in row_keys]
    fractions = sorted(pred["fraction_b"].unique())
    states = sorted(pred["openworld_state_family"].astype(str).unique())
    palette = [
        "#1565C0",
        "#D9822B",
        "#2E7D32",
        "#7B1FA2",
        "#00838F",
        "#6D4C41",
        "#C62828",
        "#455A64",
    ]
    color_map = {state: palette[i % len(palette)] for i, state in enumerate(states)}
    cell_w, cell_h = 74, 28
    left, top = 170, 62
    width = left + cell_w * len(fractions) + 250
    height = top + cell_h * len(rows) + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Open-world state trajectories across controlled mixtures</text>',
    ]
    for j, frac in enumerate(fractions):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top - 12}" font-family="Arial" font-size="11" text-anchor="middle">{frac:.2f}</text>')
    for i, (design, pair_index) in enumerate(rows):
        y = top + i * cell_h
        parts.append(f'<text x="20" y="{y + 18}" font-family="Arial" font-size="11">{html.escape(design)} pair {pair_index}</text>')
        for j, frac in enumerate(fractions):
            row = pred.loc[(pred["design"] == design) & (pred["pair_index"] == pair_index) & (pred["fraction_b"] == frac)]
            if row.empty:
                continue
            r = row.iloc[0]
            state = str(r["openworld_state_family"])
            status = str(r["openworld_disease_status"])
            x = left + j * cell_w
            stroke = "#111" if status == "stable" else ("#777" if status == "mixed" else "#bbb")
            dash = "" if status == "stable" else ' stroke-dasharray="4 3"'
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" fill="{color_map[state]}" stroke="{stroke}"{dash}/>')
            parts.append(f'<text x="{x + cell_w / 2}" y="{y + 18}" font-family="Arial" font-size="10" text-anchor="middle" fill="#fff">{html.escape(status[:1])}</text>')
    lx = left + cell_w * len(fractions) + 28
    for i, state in enumerate(states):
        y = top + i * 22
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color_map[state]}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="11">{html.escape(state)}</text>')
    parts.append('<text x="24" y="{0}" font-family="Arial" font-size="11" fill="#666">Cell text: s=stable, m=mixed, u=unsupported. Dashed borders mark non-stable disease semantics.</text>'.format(height - 22))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(pred: pd.DataFrame, frac_summary: pd.DataFrame) -> None:
    mixed = pred.loc[pred["fraction_b"].isin([0.10, 0.25, 0.50, 0.75])]
    high_over = float(mixed["closed_set_highconf_overcall_70"].mean()) if len(mixed) else float("nan")
    mean_conf = float(mixed["closed_set_confidence"].mean()) if len(mixed) else float("nan")
    open_nonstable = float(mixed["openworld_mixed_or_unsupported"].mean()) if len(mixed) else float("nan")
    lines = [
        "# Controlled in silico profile mixing",
        "",
        "## Purpose",
        "",
        "This first-pass experiment linearly mixes external bulk RNA profiles representing tumor/epithelial-proliferative, immune/hematologic, and clean/normal-like programs. Each synthetic mixture is evaluated by the closed-set disease-family baseline and the frozen open-world semantic explainer.",
        "",
        "## Design",
        "",
        f"- replicate pairs per design: `{N_REPLICATE_PAIRS}`",
        f"- fractions of component B: `{FRACTIONS}`",
        "- designs: `tumor_normal`, `tumor_immune`, `normal_immune`",
        "",
        "## Intermediate-mixture findings",
        "",
        f"- mean closed-set confidence across intermediate mixtures: `{mean_conf:.4f}`",
        f"- high-confidence closed-set overcall rate vs open-world mixed/unsupported at confidence >= 0.70: `{high_over:.4f}`",
        f"- open-world mixed/unsupported rate across intermediate mixtures: `{open_nonstable:.4f}`",
        "",
        "## Fraction-level summary",
        "",
    ]
    for row in frac_summary.itertuples(index=False):
        lines.append(
            f"- `{row.design}` f={row.fraction_b:.2f}: closed_conf={row.mean_closed_set_confidence:.3f}, "
            f"open_nonstable={row.openworld_mixed_or_unsupported_rate:.3f}, "
            f"closed={row.most_common_closed_set_disease_family}, "
            f"state={row.most_common_openworld_state_family}, status={row.most_common_openworld_disease_status}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This controlled mixing analysis tests whether intermediate mixtures receive confident predefined disease labels while the RNA-language portrait readout marks them as mixed or unsupported and changes portrait family along the trajectory.",
            "",
            "## Output files",
            "",
            "- `t5_mixing_design.csv`",
            "- `t5_mixing_predictions.csv`",
            "- `t5_fraction_summary.csv`",
            "- `t5_closedset_openworld_curve.svg`",
            "- `t5_state_trajectory.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(RNG_SEED)
    scores = pd.read_csv(T4_SCORES)
    design = build_mixing_design(scores)
    design.to_csv(OUTDIR / "t5_mixing_design.csv", index=False)
    clf, ckpt = train_closed_set_baseline()
    pred = run_predictions(design, clf, ckpt)
    frac_summary = summarize_predictions(pred)
    pred.to_csv(OUTDIR / "t5_mixing_predictions.csv", index=False)
    frac_summary.to_csv(OUTDIR / "t5_fraction_summary.csv", index=False)
    svg_closed_open_curve(frac_summary, OUTDIR / "t5_closedset_openworld_curve.svg")
    svg_state_trajectory(pred, OUTDIR / "t5_state_trajectory.svg")
    write_summary(pred, frac_summary)


if __name__ == "__main__":
    main()
