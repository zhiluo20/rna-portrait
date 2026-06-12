#!/usr/bin/env python3
"""Add calibrated closed-set disease predictions to the controlled mixing trajectories."""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

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
OUTDIR = SUPP_DIR / "T5_calibrated_mixing"

T2B_SCRIPT = SCRIPT_DIR / "03_predefined_label_calibration.py"
T5_SCRIPT = SCRIPT_DIR / "10_profile_mixing_initial.py"
T5_DIR = SUPP_DIR / "T5_controlled_mixing"
T5_DESIGN = T5_DIR / "t5_mixing_design.csv"
T5_OPENWORLD = T5_DIR / "t5_mixing_predictions.csv"

FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
THRESHOLDS = [0.5, 0.7, 0.9]


def load_module(path: Path, module_name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def train_closed_set_models():
    t2b = load_module(T2B_SCRIPT, "t5b_t2b_helpers")
    t2 = t2b.load_module(t2b.T2_SCRIPT, "t5b_t2_helpers")
    train_module = t2.load_module(CODE_DIR / "train_rna_language_alignment.py", "t5b_train_module")
    taxonomy = t2.load_module(CODE_DIR / "family_taxonomy.py", "t5b_family_taxonomy")
    ckpt = t2.torch_load(t2.CHECKPOINT)
    x_train, train_meta = t2.prepare_training_matrix(train_module, ckpt, split="train")
    x_val, val_meta = t2.prepare_training_matrix(train_module, ckpt, split="val")
    y_train, keep = t2b.make_labels_with_train_vocab(train_meta, taxonomy)
    y_val, _ = t2b.make_labels_with_train_vocab(val_meta, taxonomy, keep)
    base = t2b.train_base_classifier(x_train, y_train)
    class_order = np.asarray(base.classes_).astype(str)
    temp = t2b.fit_temperature_scaled(base, x_val, y_val, class_order)
    return t2b, ckpt, class_order, {
        "uncalibrated_sgd": base,
        "temperature_scaled_sgd": temp,
    }


def predict_model(t2b, model, x: np.ndarray, class_order: np.ndarray) -> Tuple[str, float]:
    probs = t2b.aligned_predict_proba(model, x, class_order)
    pred, conf = t2b.top_predictions(probs, class_order)
    return str(pred[0]), float(conf[0])


def build_predictions() -> pd.DataFrame:
    t2b, ckpt, class_order, models = train_closed_set_models()
    t5 = load_module(T5_SCRIPT, "t5b_t5_helpers")
    design = pd.read_csv(T5_DESIGN)
    openworld = pd.read_csv(T5_OPENWORLD)
    openworld_by_id = {str(r.synthetic_sample_id): r._asdict() for r in openworld.itertuples(index=False)}
    rows = []
    for pair_idx, rec in enumerate(design.itertuples(index=False), start=1):
        a = t5.load_profile(str(rec.component_a_pool), str(rec.component_a_file))
        b = t5.load_profile(str(rec.component_b_pool), str(rec.component_b_file))
        for frac in FRACTIONS:
            sample_id = f"{rec.design}_pair{pair_idx:02d}_f{int(round(frac * 100)):03d}"
            values = t5.mix_profiles(a, b, frac)
            x = t5.selected_gene_matrix(values, ckpt)
            ow = openworld_by_id[sample_id]
            for model_name, model in models.items():
                pred, conf = predict_model(t2b, model, x, class_order)
                row = {
                    "synthetic_sample_id": sample_id,
                    "model": model_name,
                    "temperature": float(getattr(model, "temperature", np.nan)),
                    "design": str(rec.design),
                    "pair_index": int(pair_idx),
                    "fraction_label": str(rec.fraction_label),
                    "fraction_b": float(frac),
                    "component_a_pool": str(rec.component_a_pool),
                    "component_a_file": str(rec.component_a_file),
                    "component_a_expected_disease_family": str(rec.component_a_expected_disease_family),
                    "component_b_pool": str(rec.component_b_pool),
                    "component_b_file": str(rec.component_b_file),
                    "component_b_expected_disease_family": str(rec.component_b_expected_disease_family),
                    "closed_set_disease_family": pred,
                    "closed_set_confidence": conf,
                    "closed_set_forced_stable": True,
                    "openworld_state_family": str(ow["openworld_state_family"]),
                    "openworld_state_subprofile": str(ow["openworld_state_subprofile"]),
                    "openworld_resolved_disease_family": str(ow["openworld_resolved_disease_family"]),
                    "openworld_disease_status": str(ow["openworld_disease_status"]),
                    "openworld_disease_subprofile": str(ow["openworld_disease_subprofile"]),
                    "openworld_anchor_context_state": str(ow["openworld_anchor_context_state"]),
                    "openworld_stable": bool(ow["openworld_stable"]),
                    "openworld_mixed_or_unsupported": bool(ow["openworld_mixed_or_unsupported"]),
                }
                for threshold in THRESHOLDS:
                    suffix = int(threshold * 100)
                    row[f"covered_{suffix}"] = conf >= threshold
                    row[f"highconf_overcall_{suffix}"] = (conf >= threshold) and row["openworld_mixed_or_unsupported"]
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_fraction(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, design, fraction), sub in pred.groupby(["model", "design", "fraction_b"], dropna=False):
        row = {
            "model": model,
            "design": design,
            "fraction_b": float(fraction),
            "n": int(len(sub)),
            "mean_closed_set_confidence": float(sub["closed_set_confidence"].mean()),
            "openworld_mixed_or_unsupported_rate": float(sub["openworld_mixed_or_unsupported"].mean()),
            "openworld_stable_rate": float(sub["openworld_stable"].mean()),
            "most_common_closed_set_disease_family": str(sub["closed_set_disease_family"].mode().iloc[0]),
            "most_common_openworld_state_family": str(sub["openworld_state_family"].mode().iloc[0]),
            "most_common_openworld_disease_status": str(sub["openworld_disease_status"].mode().iloc[0]),
        }
        for threshold in THRESHOLDS:
            suffix = int(threshold * 100)
            covered = sub[f"covered_{suffix}"]
            row[f"coverage_{suffix}"] = float(covered.mean())
            row[f"highconf_overcall_rate_{suffix}"] = float(sub[f"highconf_overcall_{suffix}"].mean())
            if covered.any():
                row[f"overcall_among_covered_{suffix}"] = float(sub.loc[covered, "openworld_mixed_or_unsupported"].mean())
            else:
                row[f"overcall_among_covered_{suffix}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["model", "design", "fraction_b"])


def summarize_model(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    intermediate = pred.loc[pred["fraction_b"].isin([0.10, 0.25, 0.50, 0.75])].copy()
    for model, sub in intermediate.groupby("model"):
        row = {
            "model": model,
            "n_intermediate": int(len(sub)),
            "mean_closed_set_confidence": float(sub["closed_set_confidence"].mean()),
            "openworld_mixed_or_unsupported_rate": float(sub["openworld_mixed_or_unsupported"].mean()),
            "openworld_stable_rate": float(sub["openworld_stable"].mean()),
        }
        for threshold in THRESHOLDS:
            suffix = int(threshold * 100)
            covered = sub[f"covered_{suffix}"]
            row[f"coverage_{suffix}"] = float(covered.mean())
            row[f"highconf_overcall_rate_{suffix}"] = float(sub[f"highconf_overcall_{suffix}"].mean())
            row[f"overcall_among_covered_{suffix}"] = (
                float(sub.loc[covered, "openworld_mixed_or_unsupported"].mean()) if covered.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def svg_confidence_trajectory(frac: pd.DataFrame, path: Path) -> None:
    width, height = 980, 430
    left, top, plot_w, plot_h = 78, 60, 710, 280
    colors = {"uncalibrated_sgd": "#1565C0", "temperature_scaled_sgd": "#7B1FA2"}
    dash = {"tumor_normal": "", "tumor_immune": "7 5", "normal_immune": "3 4"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Calibrated closed-set confidence across controlled mixtures</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    def xmap(v: float) -> float:
        return left + float(v) * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - float(v) * plot_h

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{xmap(tick):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.2f}</text>')
        parts.append(f'<text x="{left - 12}" y="{ymap(tick) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for (model, design), sub in frac.groupby(["model", "design"]):
        sub = sub.sort_values("fraction_b")
        pts = [f"{xmap(r.fraction_b):.1f},{ymap(r.mean_closed_set_confidence):.1f}" for r in sub.itertuples(index=False)]
        stroke_dash = f' stroke-dasharray="{dash.get(design, "")}"' if dash.get(design, "") else ""
        parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{colors.get(model, "#555")}" stroke-width="3"{stroke_dash}/>')
        for r in sub.itertuples(index=False):
            parts.append(f'<circle cx="{xmap(r.fraction_b):.1f}" cy="{ymap(r.mean_closed_set_confidence):.1f}" r="4" fill="{colors.get(model, "#555")}"/>')
    lx = width - 190
    for i, (model, color) in enumerate(colors.items()):
        y = 76 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(model)}</text>')
    for i, (design, pattern) in enumerate(dash.items()):
        y = 146 + i * 24
        dash_attr = f' stroke-dasharray="{pattern}"' if pattern else ""
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx + 34}" y2="{y}" stroke="#555" stroke-width="3"{dash_attr}/>')
        parts.append(f'<text x="{lx + 42}" y="{y + 4}" font-family="Arial" font-size="12">{html.escape(design)}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" font-family="Arial" font-size="13" text-anchor="middle">fraction of component B</text>')
    parts.append('<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">closed-set confidence</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_overcall_coverage(model_summary: pd.DataFrame, path: Path) -> None:
    width, height = 820, 420
    left, top, plot_w, plot_h = 86, 64, 520, 265
    colors = {"uncalibrated_sgd": "#1565C0", "temperature_scaled_sgd": "#7B1FA2"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">High-confidence overcall after calibration</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    def xmap(v: float) -> float:
        return left + (v - 0.5) / 0.4 * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - float(v) * plot_h

    for tick in THRESHOLDS:
        parts.append(f'<text x="{xmap(tick):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.1f}</text>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{left - 12}" y="{ymap(tick) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for _, row in model_summary.iterrows():
        model = str(row["model"])
        color = colors.get(model, "#555")
        pts = []
        for threshold in THRESHOLDS:
            suffix = int(threshold * 100)
            v = row[f"overcall_among_covered_{suffix}"]
            if pd.isna(v):
                continue
            pts.append(f"{xmap(threshold):.1f},{ymap(v):.1f}")
            parts.append(f'<circle cx="{xmap(threshold):.1f}" cy="{ymap(v):.1f}" r="5" fill="{color}"/>')
            parts.append(f'<text x="{xmap(threshold):.1f}" y="{ymap(v) - 8:.1f}" font-family="Arial" font-size="10" text-anchor="middle">{v:.2f}</text>')
        if pts:
            parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
    lx = width - 198
    for i, (model, color) in enumerate(colors.items()):
        y = 80 + i * 26
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(model)}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" font-family="Arial" font-size="13" text-anchor="middle">confidence threshold</text>')
    parts.append('<text x="20" y="218" font-family="Arial" font-size="13" transform="rotate(-90 20 218)" text-anchor="middle">overcall among covered intermediate mixtures</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(model_summary: pd.DataFrame, fraction_summary: pd.DataFrame) -> None:
    lines = [
        "# Calibrated controlled mixing trajectory",
        "",
        "## Purpose",
        "",
        "This analysis reuses the controlled in silico mixtures and replaces the predefined-label confidence with the temperature-scaled predefined disease-label baseline. The portrait trajectory is unchanged; only the predefined-label confidence is recalculated.",
        "",
        "## Intermediate mixtures",
        "",
    ]
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"- `{row.model}`: mean confidence `{row.mean_closed_set_confidence:.4f}`, open-world mixed/unsupported `{row.openworld_mixed_or_unsupported_rate:.4f}`, "
            f"coverage>=0.70 `{row.coverage_70:.4f}`, overcall among covered>=0.70 `{row.overcall_among_covered_70 if not pd.isna(row.overcall_among_covered_70) else 'NA'}`"
        )
    lines.extend(["", "## Fraction-level trajectory", ""])
    temp = fraction_summary.loc[fraction_summary["model"] == "temperature_scaled_sgd"]
    for row in temp.itertuples(index=False):
        over = "NA" if pd.isna(row.overcall_among_covered_70) else f"{row.overcall_among_covered_70:.3f}"
        lines.append(
            f"- `{row.design}` f={row.fraction_b:.2f}: conf={row.mean_closed_set_confidence:.3f}, "
            f"coverage>=0.70={row.coverage_70:.3f}, overcall among covered>=0.70={over}, "
            f"open_nonstable={row.openworld_mixed_or_unsupported_rate:.3f}, state={row.most_common_openworld_state_family}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Temperature scaling removes the artificial all-1.0 confidence seen in the uncalibrated run while preserving the same predefined-label argmax. The key question is whether the high-confidence covered subset still mostly corresponds to mixed or unsupported portrait states.",
            "",
            "## Output files",
            "",
            "- `t5b_calibrated_mixing_predictions.csv`",
            "- `t5b_fraction_summary.csv`",
            "- `t5b_model_summary.csv`",
            "- `t5b_confidence_trajectory.svg`",
            "- `t5b_overcall_coverage.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred = build_predictions()
    fraction_summary = summarize_fraction(pred)
    model_summary = summarize_model(pred)
    pred.to_csv(OUTDIR / "t5b_calibrated_mixing_predictions.csv", index=False)
    fraction_summary.to_csv(OUTDIR / "t5b_fraction_summary.csv", index=False)
    model_summary.to_csv(OUTDIR / "t5b_model_summary.csv", index=False)
    svg_confidence_trajectory(fraction_summary, OUTDIR / "t5b_confidence_trajectory.svg")
    svg_overcall_coverage(model_summary, OUTDIR / "t5b_overcall_coverage.svg")
    write_summary(model_summary, fraction_summary)


if __name__ == "__main__":
    main()
