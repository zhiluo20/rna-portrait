#!/usr/bin/env python3
"""Expanded calibrated controlled mixing with bootstrap confidence intervals."""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
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
OUTDIR = SUPP_DIR / "T5c_expanded_calibrated_mixing_bootstrap"

T5_SCRIPT = SCRIPT_DIR / "10_profile_mixing_initial.py"
T5B_SCRIPT = SCRIPT_DIR / "11_profile_mixing_calibrated.py"
T4_SCORES = SUPP_DIR / "T4_marker_program_state_validation" / "t4_marker_scores_by_sample.csv"

N_REPLICATE_PAIRS = 10
FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
INTERMEDIATE_FRACTIONS = [0.10, 0.25, 0.50, 0.75]
THRESHOLDS = [0.5, 0.7, 0.9]
N_BOOTSTRAP = 1000
RNG_SEED = 20260604


def load_module(path: Path, module_name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def top_unique(df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
    # The two external pools can contain the same source sample. For expanded
    # mixing, count each source file only once so replicate pairs are independent.
    return (
        df.sort_values(score_col, ascending=False)
        .drop_duplicates(["file"])
        .head(n)
        .reset_index(drop=True)
    )


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


def select_component_samples(t5, scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = t5.add_selection_scores(scores)
    disease = df["expected_disease_family"].astype(str)
    site = df["expected_site_family"].astype(str)
    state = df["semantic_state_family"].astype(str)
    status = df["semantic_disease_semantic_status"].astype(str)

    tumor_mask = disease.str.contains("tumor|malignancy", case=False, regex=True)
    tumor_mask &= ~disease.str.contains("non_tumor|healthy_control", case=False, regex=True)
    tumor = top_unique(df.loc[tumor_mask], "tumor_program_score", N_REPLICATE_PAIRS)

    immune_mask = (site == "hematologic") | state.str.contains("hematologic|immune", case=False, regex=True)
    immune = top_unique(df.loc[immune_mask], "immune_program_score", N_REPLICATE_PAIRS)

    normal_mask = (disease == "healthy_control") & (status == "stable")
    normal = top_unique(df.loc[normal_mask], "clean_program_score", N_REPLICATE_PAIRS)
    if len(normal) < N_REPLICATE_PAIRS:
        fallback = disease.isin(["healthy_control", "other_non_tumor"]) & (status != "unsupported")
        normal = top_unique(df.loc[fallback], "clean_program_score", N_REPLICATE_PAIRS)

    if min(len(tumor), len(immune), len(normal)) < N_REPLICATE_PAIRS:
        raise RuntimeError(
            f"not enough component samples: tumor={len(tumor)}, immune={len(immune)}, normal={len(normal)}"
        )
    return tumor, immune, normal


def build_mixing_design(t5, scores: pd.DataFrame) -> pd.DataFrame:
    tumor, immune, normal = select_component_samples(t5, scores)
    rows = []
    for i in range(N_REPLICATE_PAIRS):
        rank = i + 1
        t = tumor.iloc[i]
        im = immune.iloc[i]
        n = normal.iloc[i]
        rows.append(
            {
                "design": "tumor_normal",
                "replicate_id": rank,
                "fraction_label": "tumor_fraction",
                **component_row(n, "component_a", rank),
                **component_row(t, "component_b", rank),
            }
        )
        rows.append(
            {
                "design": "tumor_immune",
                "replicate_id": rank,
                "fraction_label": "immune_fraction",
                **component_row(t, "component_a", rank),
                **component_row(im, "component_b", rank),
            }
        )
        rows.append(
            {
                "design": "normal_immune",
                "replicate_id": rank,
                "fraction_label": "immune_fraction",
                **component_row(n, "component_a", rank),
                **component_row(im, "component_b", rank),
            }
        )
    return pd.DataFrame(rows)


def train_closed_set_models(t5b):
    t2b, ckpt, class_order, models = t5b.train_closed_set_models()
    return t2b, ckpt, class_order, models


def build_predictions(t5, t5b, design: pd.DataFrame) -> pd.DataFrame:
    t2b, ckpt, class_order, models = train_closed_set_models(t5b)
    explainer = t5.load_module(t5.EXPLAINER_SCRIPT, "t5c_semantic_explainer")
    rows = []
    for rec in design.itertuples(index=False):
        a = t5.load_profile(str(rec.component_a_pool), str(rec.component_a_file))
        b = t5.load_profile(str(rec.component_b_pool), str(rec.component_b_file))
        for frac in FRACTIONS:
            sample_id = f"{rec.design}_rep{int(rec.replicate_id):02d}_f{int(round(frac * 100)):03d}"
            values = t5.mix_profiles(a, b, frac)
            x = t5.selected_gene_matrix(values, ckpt)
            counts = {str(g): float(v) for g, v in values.items() if np.isfinite(v)}
            payload = explainer.explain_counts(counts, top_k=5, rerank_beta=0.3)
            openworld = t5.extract_openworld(payload)
            openworld["openworld_stable"] = openworld["openworld_disease_status"] == "stable"
            openworld["openworld_mixed_or_unsupported"] = openworld["openworld_disease_status"] != "stable"
            for model_name, model in models.items():
                pred, conf = t5b.predict_model(t2b, model, x, class_order)
                row = {
                    "synthetic_sample_id": sample_id,
                    "model": str(model_name),
                    "temperature": float(getattr(model, "temperature", np.nan)),
                    "design": str(rec.design),
                    "replicate_id": int(rec.replicate_id),
                    "fraction_label": str(rec.fraction_label),
                    "fraction_b": float(frac),
                    "component_a_pool": str(rec.component_a_pool),
                    "component_a_file": str(rec.component_a_file),
                    "component_a_expected_disease_family": str(rec.component_a_expected_disease_family),
                    "component_a_expected_site_family": str(rec.component_a_expected_site_family),
                    "component_a_semantic_state_family": str(rec.component_a_semantic_state_family),
                    "component_b_pool": str(rec.component_b_pool),
                    "component_b_file": str(rec.component_b_file),
                    "component_b_expected_disease_family": str(rec.component_b_expected_disease_family),
                    "component_b_expected_site_family": str(rec.component_b_expected_site_family),
                    "component_b_semantic_state_family": str(rec.component_b_semantic_state_family),
                    "closed_set_disease_family": str(pred),
                    "closed_set_confidence": float(conf),
                    "closed_set_forced_stable": True,
                    **openworld,
                }
                for threshold in THRESHOLDS:
                    suffix = int(threshold * 100)
                    row[f"covered_{suffix}"] = row["closed_set_confidence"] >= threshold
                    row[f"highconf_overcall_{suffix}"] = (
                        row["closed_set_confidence"] >= threshold and row["openworld_mixed_or_unsupported"]
                    )
                rows.append(row)
            print(
                f"[profile-mixing-bootstrap] {sample_id}: portrait={openworld['openworld_disease_status']}/"
                f"{openworld['openworld_state_family']}",
                flush=True,
            )
    return pd.DataFrame(rows)


def overcall_among_covered(sub: pd.DataFrame, suffix: int) -> float:
    covered = sub[f"covered_{suffix}"].astype(bool)
    if not covered.any():
        return np.nan
    return float(sub.loc[covered, "openworld_mixed_or_unsupported"].mean())


def metric_dict(sub: pd.DataFrame) -> Dict[str, float]:
    row = {
        "n": int(len(sub)),
        "mean_closed_set_confidence": float(sub["closed_set_confidence"].mean()),
        "openworld_mixed_or_unsupported_rate": float(sub["openworld_mixed_or_unsupported"].mean()),
        "openworld_stable_rate": float(sub["openworld_stable"].mean()),
    }
    for threshold in THRESHOLDS:
        suffix = int(threshold * 100)
        row[f"coverage_{suffix}"] = float(sub[f"covered_{suffix}"].mean())
        row[f"highconf_overcall_rate_{suffix}"] = float(sub[f"highconf_overcall_{suffix}"].mean())
        row[f"overcall_among_covered_{suffix}"] = overcall_among_covered(sub, suffix)
    return row


def summarize_fraction(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, design, fraction), sub in pred.groupby(["model", "design", "fraction_b"], dropna=False):
        row = {
            "model": str(model),
            "design": str(design),
            "fraction_b": float(fraction),
            **metric_dict(sub),
            "most_common_closed_set_disease_family": str(sub["closed_set_disease_family"].mode().iloc[0]),
            "most_common_openworld_state_family": str(sub["openworld_state_family"].mode().iloc[0]),
            "most_common_openworld_disease_status": str(sub["openworld_disease_status"].mode().iloc[0]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "design", "fraction_b"])


def summarize_model(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intermediate = pred.loc[pred["fraction_b"].isin(INTERMEDIATE_FRACTIONS)].copy()
    for model, sub in intermediate.groupby("model"):
        rows.append({"model": str(model), **metric_dict(sub)})
    return pd.DataFrame(rows).sort_values("model")


def quantile_ci(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def bootstrap_fraction_ci(pred: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    metrics = [
        "mean_closed_set_confidence",
        "openworld_mixed_or_unsupported_rate",
        "coverage_70",
        "highconf_overcall_rate_70",
        "overcall_among_covered_70",
    ]
    for (model, design), sub in pred.groupby(["model", "design"], dropna=False):
        reps = sorted(sub["replicate_id"].unique())
        for fraction in sorted(sub["fraction_b"].unique()):
            frac_sub = sub.loc[sub["fraction_b"] == fraction]
            boot_vals = {m: [] for m in metrics}
            for _ in range(N_BOOTSTRAP):
                sampled = rng.choice(reps, size=len(reps), replace=True)
                sample = pd.concat([frac_sub.loc[frac_sub["replicate_id"] == rep] for rep in sampled], ignore_index=True)
                vals = metric_dict(sample)
                for metric in metrics:
                    boot_vals[metric].append(vals[metric])
            row = {"model": str(model), "design": str(design), "fraction_b": float(fraction)}
            for metric in metrics:
                lo, hi = quantile_ci(boot_vals[metric])
                row[f"{metric}_lo"] = lo
                row[f"{metric}_hi"] = hi
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "design", "fraction_b"])


def bootstrap_model_ci(pred: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    metrics = [
        "mean_closed_set_confidence",
        "openworld_mixed_or_unsupported_rate",
        "coverage_70",
        "highconf_overcall_rate_70",
        "overcall_among_covered_70",
    ]
    intermediate = pred.loc[pred["fraction_b"].isin(INTERMEDIATE_FRACTIONS)].copy()
    for model, sub in intermediate.groupby("model"):
        units = sorted(set(zip(sub["design"].astype(str), sub["replicate_id"].astype(int))))
        boot_vals = {m: [] for m in metrics}
        for _ in range(N_BOOTSTRAP):
            sampled = [units[i] for i in rng.integers(0, len(units), size=len(units))]
            sample_parts = [
                sub.loc[(sub["design"] == design) & (sub["replicate_id"] == rep)]
                for design, rep in sampled
            ]
            sample = pd.concat(sample_parts, ignore_index=True)
            vals = metric_dict(sample)
            for metric in metrics:
                boot_vals[metric].append(vals[metric])
        row = {"model": str(model)}
        for metric in metrics:
            lo, hi = quantile_ci(boot_vals[metric])
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model")


def merge_ci(point: pd.DataFrame, ci: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    return point.merge(ci, on=keys, how="left")


def xmap(v: float, left: float, plot_w: float) -> float:
    return left + float(v) * plot_w


def ymap(v: float, top: float, plot_h: float) -> float:
    return top + plot_h - float(v) * plot_h


def svg_temperature_confidence_ci(frac: pd.DataFrame, path: Path) -> None:
    sub = frac.loc[frac["model"] == "temperature_scaled_sgd"].copy()
    width, height = 980, 430
    left, top, plot_w, plot_h = 78, 60, 710, 280
    colors = {"tumor_normal": "#1565C0", "tumor_immune": "#D9822B", "normal_immune": "#2E7D32"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Expanded T5c calibrated confidence with bootstrap CI</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{xmap(tick, left, plot_w):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.2f}</text>')
        parts.append(f'<text x="{left - 12}" y="{ymap(tick, top, plot_h) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for design, dsub in sub.groupby("design"):
        dsub = dsub.sort_values("fraction_b")
        color = colors.get(str(design), "#555")
        pts = []
        for row in dsub.itertuples(index=False):
            x = xmap(row.fraction_b, left, plot_w)
            y = ymap(row.mean_closed_set_confidence, top, plot_h)
            lo = ymap(row.mean_closed_set_confidence_lo, top, plot_h)
            hi = ymap(row.mean_closed_set_confidence_hi, top, plot_h)
            pts.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<line x1="{x:.1f}" y1="{hi:.1f}" x2="{x:.1f}" y2="{lo:.1f}" stroke="{color}" stroke-width="2" opacity="0.45"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
    lx = width - 170
    for i, (design, color) in enumerate(colors.items()):
        y = 78 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(design)}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" font-family="Arial" font-size="13" text-anchor="middle">fraction of component B</text>')
    parts.append('<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">temperature-scaled confidence</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_model_bootstrap_summary(model: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("coverage_70", "coverage >=0.70"),
        ("overcall_among_covered_70", "overcall among covered"),
        ("openworld_mixed_or_unsupported_rate", "open mixed/unsupported"),
    ]
    width, height = 860, 430
    left, top, plot_w, plot_h = 96, 62, 560, 260
    bar_w = 44
    colors = {"uncalibrated_sgd": "#1565C0", "temperature_scaled_sgd": "#7B1FA2"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Expanded T5c intermediate-mixture bootstrap summary</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{left - 12}" y="{ymap(tick, top, plot_h) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for i, (metric, label) in enumerate(metrics):
        group_x = left + 90 + i * 170
        parts.append(f'<text x="{group_x + 22}" y="{top + plot_h + 35}" font-family="Arial" font-size="11" text-anchor="middle">{html.escape(label)}</text>')
        for j, row in enumerate(model.itertuples(index=False)):
            model_name = str(row.model)
            val = float(getattr(row, metric))
            lo = float(getattr(row, f"{metric}_lo"))
            hi = float(getattr(row, f"{metric}_hi"))
            x = group_x + j * (bar_w + 8)
            y = ymap(val, top, plot_h)
            ylo = ymap(lo, top, plot_h)
            yhi = ymap(hi, top, plot_h)
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{top + plot_h - y:.1f}" fill="{colors.get(model_name, "#555")}" opacity="0.82"/>')
            parts.append(f'<line x1="{x + bar_w / 2:.1f}" y1="{yhi:.1f}" x2="{x + bar_w / 2:.1f}" y2="{ylo:.1f}" stroke="#222" stroke-width="2"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" font-family="Arial" font-size="10" text-anchor="middle">{val:.2f}</text>')
    lx = width - 212
    for i, (model_name, color) in enumerate(colors.items()):
        y = 78 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(model_name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(model_summary: pd.DataFrame, fraction_summary: pd.DataFrame) -> None:
    lines = [
        "# Expanded calibrated controlled mixing with bootstrap intervals",
        "",
        "## Purpose",
        "",
        "This analysis expands the controlled-mixing workflow from 3 to 10 replicate pairs per design, fixes the component-selection rule so non-tumor samples cannot enter the tumor component, and adds pair-level bootstrap confidence intervals.",
        "",
        "## Design",
        "",
        f"- replicate pairs per design: `{N_REPLICATE_PAIRS}`",
        f"- designs: `tumor_normal`, `tumor_immune`, `normal_immune`",
        f"- fractions: `{FRACTIONS}`",
        f"- bootstrap iterations: `{N_BOOTSTRAP}`",
        "",
        "## Intermediate-mixture findings",
        "",
    ]
    for row in model_summary.itertuples(index=False):
        over = "NA" if pd.isna(row.overcall_among_covered_70) else f"{row.overcall_among_covered_70:.4f}"
        over_ci = (
            "NA"
            if pd.isna(row.overcall_among_covered_70_lo)
            else f"{row.overcall_among_covered_70_lo:.4f}-{row.overcall_among_covered_70_hi:.4f}"
        )
        lines.append(
            f"- `{row.model}`: mean confidence `{row.mean_closed_set_confidence:.4f}` "
            f"[{row.mean_closed_set_confidence_lo:.4f}, {row.mean_closed_set_confidence_hi:.4f}], "
            f"open-world mixed/unsupported `{row.openworld_mixed_or_unsupported_rate:.4f}` "
            f"[{row.openworld_mixed_or_unsupported_rate_lo:.4f}, {row.openworld_mixed_or_unsupported_rate_hi:.4f}], "
            f"coverage>=0.70 `{row.coverage_70:.4f}` [{row.coverage_70_lo:.4f}, {row.coverage_70_hi:.4f}], "
            f"overcall among covered>=0.70 `{over}` [{over_ci}]"
        )
    lines.extend(["", "## Temperature-scaled fraction-level trajectory", ""])
    temp = fraction_summary.loc[fraction_summary["model"] == "temperature_scaled_sgd"]
    for row in temp.itertuples(index=False):
        over = "NA" if pd.isna(row.overcall_among_covered_70) else f"{row.overcall_among_covered_70:.3f}"
        lines.append(
            f"- `{row.design}` f={row.fraction_b:.2f}: conf={row.mean_closed_set_confidence:.3f} "
            f"[{row.mean_closed_set_confidence_lo:.3f}, {row.mean_closed_set_confidence_hi:.3f}], "
            f"open_nonstable={row.openworld_mixed_or_unsupported_rate:.3f}, "
            f"coverage>=0.70={row.coverage_70:.3f}, overcall among covered>=0.70={over}, "
            f"state={row.most_common_openworld_state_family}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The expanded run reports the controlled-mixing analysis with replicate profile pairs and bootstrap intervals. It evaluates whether predefined disease-family models assign single labels to intermediate mixtures, while the RNA-language portrait readout marks a fraction of mixtures as mixed or unsupported. The covered subset of calibrated predefined-label calls is also audited for overcall.",
            "",
            "## Output files",
            "",
            "- `t5c_mixing_design.csv`",
            "- `t5c_expanded_predictions.csv`",
            "- `t5c_fraction_summary.csv`",
            "- `t5c_model_summary.csv`",
            "- `t5c_fraction_bootstrap_ci.csv`",
            "- `t5c_model_bootstrap_ci.csv`",
            "- `t5c_temperature_confidence_ci.svg`",
            "- `t5c_model_bootstrap_summary.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t5 = load_module(T5_SCRIPT, "t5c_t5_helpers")
    t5b = load_module(T5B_SCRIPT, "t5c_t5b_helpers")
    scores = pd.read_csv(T4_SCORES)
    design = build_mixing_design(t5, scores)
    design.to_csv(OUTDIR / "t5c_mixing_design.csv", index=False)
    pred = build_predictions(t5, t5b, design)
    fraction = summarize_fraction(pred)
    model = summarize_model(pred)
    frac_ci = bootstrap_fraction_ci(pred)
    model_ci = bootstrap_model_ci(pred)
    fraction_out = merge_ci(fraction, frac_ci, ["model", "design", "fraction_b"])
    model_out = merge_ci(model, model_ci, ["model"])

    pred.to_csv(OUTDIR / "t5c_expanded_predictions.csv", index=False)
    fraction_out.to_csv(OUTDIR / "t5c_fraction_summary.csv", index=False)
    model_out.to_csv(OUTDIR / "t5c_model_summary.csv", index=False)
    frac_ci.to_csv(OUTDIR / "t5c_fraction_bootstrap_ci.csv", index=False)
    model_ci.to_csv(OUTDIR / "t5c_model_bootstrap_ci.csv", index=False)
    svg_temperature_confidence_ci(fraction_out, OUTDIR / "t5c_temperature_confidence_ci.svg")
    svg_model_bootstrap_summary(model_out, OUTDIR / "t5c_model_bootstrap_summary.svg")
    write_summary(model_out, fraction_out)


if __name__ == "__main__":
    main()
