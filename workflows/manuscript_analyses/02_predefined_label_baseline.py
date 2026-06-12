#!/usr/bin/env python3
"""Train a closed-set disease-family baseline and compare against open-world outputs."""

from __future__ import annotations

import html
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


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
OUTDIR = SUPP_DIR / "T2_closed_set_disease_overcall"

CHECKPOINT = BACKBONE_DIR / "bulk_multimodal_embedding.pt"
SAMPLE_EMBEDDINGS = BACKBONE_DIR / "sample_embeddings.csv"
RNG_SEED = 20260604
MIN_TRAIN_CLASS_N = 10
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]

POOLS = {
    "External-180": {
        "details": ARTIFACT_DIR / "external_180_trimmed_benchmark" / "details.csv",
        "inputs": VALIDATION_DIR / "external_180",
    },
    "MultiSource-450": {
        "details": ARTIFACT_DIR / "multisource_450_trimmed_benchmark" / "details.csv",
        "inputs": VALIDATION_DIR / "multisource_450",
    },
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


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
        gene = gene.strip().upper()
        try:
            val = float(value)
        except ValueError:
            continue
        rows.append((gene, val))
    if not rows:
        raise ValueError(f"no gene-value rows parsed from {path}")
    series = pd.DataFrame(rows, columns=["gene", "value"]).groupby("gene")["value"].sum()
    return series.astype(np.float32)


def transform_external_values(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=np.float32)
    frac_nonint = float(np.mean(np.abs(arr - np.round(arr)) > 1e-6))
    vmax = float(np.max(arr)) if len(arr) else 0.0
    if vmax > 50 or frac_nonint < 0.1:
        return np.log1p(values.clip(lower=0)).astype(np.float32)
    return values.astype(np.float32)


def make_training_labels(meta: pd.DataFrame, taxonomy) -> pd.Series:
    labels = meta.apply(
        lambda r: taxonomy.disease_family(r.get("feat_disease_label"), r.get("feat_anatomical_site"), r.get("feat_tumor_status")),
        axis=1,
    ).astype(str)
    counts = labels.value_counts()
    keep = set(counts[counts >= MIN_TRAIN_CLASS_N].index)
    return labels.where(labels.isin(keep), "rare_or_unmapped")


def load_split_assignments() -> Dict[str, str]:
    samples = pd.read_csv(SAMPLE_EMBEDDINGS, usecols=["sample_id", "split"])
    return dict(zip(samples["sample_id"].astype(str), samples["split"].astype(str)))


def prepare_training_matrix(module, ckpt: dict, split: str = "train") -> Tuple[np.ndarray, pd.DataFrame]:
    bundle = module.load_dataset()
    meta = bundle.meta.copy()
    split_assignments = load_split_assignments()
    meta["split"] = meta["sample_id"].astype(str).map(split_assignments)
    sub = meta.loc[meta["split"] == split].copy().reset_index(drop=True)
    genes = [str(g) for g in ckpt["selected_genes"]]
    expr = module.load_expr_subset(bundle, genes, sub["sample_id"].astype(str).tolist())
    x = expr.T.to_numpy(dtype=np.float32)
    mean = np.asarray(ckpt["expr_mean"], dtype=np.float32)
    std = np.asarray(ckpt["expr_std"], dtype=np.float32)
    std = np.where(std < 1e-3, 1.0, std)
    x = np.clip((x - mean) / std, -8.0, 8.0)
    return x, sub


def prepare_external_matrix(details: pd.DataFrame, input_dir: Path, ckpt: dict) -> Tuple[np.ndarray, pd.DataFrame]:
    genes = [str(g).upper() for g in ckpt["selected_genes"]]
    mean = np.asarray(ckpt["expr_mean"], dtype=np.float32)
    std = np.asarray(ckpt["expr_std"], dtype=np.float32)
    std = np.where(std < 1e-3, 1.0, std)
    rows = []
    meta_rows = []
    for rec in details.itertuples(index=False):
        file_name = str(getattr(rec, "file"))
        path = input_dir / file_name
        values = transform_external_values(parse_gene_value_file(path))
        selected = pd.Series(0.0, index=genes, dtype=np.float32)
        overlap = values.index.intersection(selected.index)
        selected.loc[overlap] = values.loc[overlap].astype(np.float32)
        x = np.clip((selected.to_numpy(dtype=np.float32) - mean) / std, -8.0, 8.0)
        rows.append(x)
        row = rec._asdict()
        row["matched_selected_genes_baseline"] = int(len(overlap))
        row["selected_gene_coverage_baseline"] = float(len(overlap) / max(1, len(genes)))
        meta_rows.append(row)
    return np.vstack(rows).astype(np.float32), pd.DataFrame(meta_rows)


def bootstrap_rate(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = float(values[idx].mean())
    return float(values.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def evaluate_pool(clf: SGDClassifier, x: np.ndarray, details: pd.DataFrame, pool: str, rng: np.random.Generator) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    scores = clf.decision_function(x)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    idx = probs.argmax(axis=1)
    pred = clf.classes_[idx]
    conf = probs[np.arange(len(idx)), idx]
    out = details.copy()
    out.insert(0, "pool", pool)
    out["closed_set_disease_family"] = pred
    out["closed_set_confidence"] = conf
    out["closed_set_expected_match"] = out["closed_set_disease_family"].astype(str) == out["expected_disease_family"].astype(str)
    out["openworld_status"] = out["semantic_disease_semantic_status"].astype(str)
    out["openworld_stable"] = out["openworld_status"] == "stable"
    out["closed_set_forced_stable"] = True
    out["closed_set_over_openworld_mixed_or_unsupported"] = ~out["openworld_stable"]
    out["closed_set_highconf_overcall_50"] = (out["closed_set_confidence"] >= 0.5) & out["closed_set_over_openworld_mixed_or_unsupported"]
    out["closed_set_highconf_overcall_70"] = (out["closed_set_confidence"] >= 0.7) & out["closed_set_over_openworld_mixed_or_unsupported"]

    closed_acc, closed_lo, closed_hi = bootstrap_rate(out["closed_set_expected_match"].to_numpy(float), rng)
    open_acc, open_lo, open_hi = bootstrap_rate(out["disease_agree_top1"].to_numpy(float), rng)
    stable_rate, stable_lo, stable_hi = bootstrap_rate(out["openworld_stable"].to_numpy(float), rng)
    high50, high50_lo, high50_hi = bootstrap_rate(out["closed_set_highconf_overcall_50"].to_numpy(float), rng)
    high70, high70_lo, high70_hi = bootstrap_rate(out["closed_set_highconf_overcall_70"].to_numpy(float), rng)
    summary = {
        "pool": pool,
        "n": int(len(out)),
        "closed_set_expected_match": closed_acc,
        "closed_set_expected_match_ci_low": closed_lo,
        "closed_set_expected_match_ci_high": closed_hi,
        "openworld_disease_agree_top1": open_acc,
        "openworld_disease_agree_top1_ci_low": open_lo,
        "openworld_disease_agree_top1_ci_high": open_hi,
        "closed_set_forced_stable_rate": 1.0,
        "openworld_stable_rate": stable_rate,
        "openworld_stable_rate_ci_low": stable_lo,
        "openworld_stable_rate_ci_high": stable_hi,
        "openworld_mixed_or_unsupported_rate": float((~out["openworld_stable"]).mean()),
        "closed_set_highconf_overcall_50": high50,
        "closed_set_highconf_overcall_50_ci_low": high50_lo,
        "closed_set_highconf_overcall_50_ci_high": high50_hi,
        "closed_set_highconf_overcall_70": high70,
        "closed_set_highconf_overcall_70_ci_low": high70_lo,
        "closed_set_highconf_overcall_70_ci_high": high70_hi,
        "mean_closed_set_confidence": float(out["closed_set_confidence"].mean()),
    }

    threshold_rows = []
    for threshold in CONF_THRESHOLDS:
        covered = out["closed_set_confidence"] >= threshold
        n_covered = int(covered.sum())
        if n_covered:
            overcall = float((covered & out["closed_set_over_openworld_mixed_or_unsupported"]).sum() / n_covered)
            expected_match = float(out.loc[covered, "closed_set_expected_match"].mean())
        else:
            overcall = np.nan
            expected_match = np.nan
        threshold_rows.append(
            {
                "pool": pool,
                "confidence_threshold": threshold,
                "coverage": float(covered.mean()),
                "n_covered": n_covered,
                "overcall_among_covered_vs_openworld": overcall,
                "expected_match_among_covered": expected_match,
            }
        )
    return out, summary, pd.DataFrame(threshold_rows)


def write_summary(summary_df: pd.DataFrame, internal: dict) -> None:
    lines = [
        "# Predefined disease-label baseline",
        "",
        "## Purpose",
        "",
        "This analysis trains a closed-set disease-family classifier on the frozen Training-25k backbone selected genes, then applies it to External-180 and MultiSource-450. The goal is to quantify what happens when every external bulk RNA profile is forced into a stable disease family.",
        "",
        "## Internal test sanity check",
        "",
        f"- Internal held-out disease-family accuracy: `{internal['test_accuracy']:.4f}`",
        f"- Training classes: `{internal['n_classes']}`",
        f"- Training samples: `{internal['n_train']}`",
        f"- Internal test samples: `{internal['n_test']}`",
        "",
        "## External findings",
        "",
    ]
    for row in summary_df.itertuples(index=False):
        lines.extend(
            [
                f"### {row.pool}",
                "",
                f"- n: `{row.n}`",
                f"- closed-set expected disease-family match: `{row.closed_set_expected_match:.4f}` (95% CI `{row.closed_set_expected_match_ci_low:.4f}`-`{row.closed_set_expected_match_ci_high:.4f}`)",
                f"- open-world disease agreement top1: `{row.openworld_disease_agree_top1:.4f}` (95% CI `{row.openworld_disease_agree_top1_ci_low:.4f}`-`{row.openworld_disease_agree_top1_ci_high:.4f}`)",
                f"- closed-set forced stable assignment rate: `1.0000`",
                f"- open-world stable disease semantics rate: `{row.openworld_stable_rate:.4f}`",
                f"- closed-set high-confidence overcall vs open-world mixed/unsupported at confidence >= 0.5: `{row.closed_set_highconf_overcall_50:.4f}`",
                f"- closed-set high-confidence overcall vs open-world mixed/unsupported at confidence >= 0.7: `{row.closed_set_highconf_overcall_70:.4f}`",
                f"- mean closed-set confidence: `{row.mean_closed_set_confidence:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "This first-pass closed-set baseline is intentionally simple. It shows the contrast between forced stable disease-family assignment and the open-world disease card, which often resolves the same external samples as mixed or unsupported. These results should be interpreted as model-comparison evidence, not yet independent biological proof; T4 biological validation is still required.",
            "",
            "## Output files",
            "",
            "- `t2_closed_set_sample_predictions.csv`",
            "- `t2_pool_summary.csv`",
            "- `t2_confidence_thresholds.csv`",
            "- `t2_prediction_status_transition.csv`",
            "- `t2_model_comparison.svg`",
            "- `t2_confidence_overcall_curve.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def svg_model_comparison(summary_df: pd.DataFrame, path: Path) -> None:
    width, height = 920, 390
    left, top, plot_h = 76, 58, 260
    group_w = 330
    colors = {"closed": "#1565C0", "open": "#D9822B", "stable": "#2E7D32"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Closed-set disease classifier forces stable labels on open-world samples</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 36}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for gi, row in enumerate(summary_df.itertuples(index=False)):
        x0 = left + 74 + gi * group_w
        vals = [
            ("closed match", row.closed_set_expected_match, colors["closed"]),
            ("open match", row.openworld_disease_agree_top1, colors["open"]),
            ("open stable", row.openworld_stable_rate, colors["stable"]),
        ]
        for i, (label, value, color) in enumerate(vals):
            h = float(value) * plot_h
            x = x0 + i * 58
            parts.append(f'<rect x="{x}" y="{top + plot_h - h:.1f}" width="38" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + 19}" y="{top + plot_h - h - 6:.1f}" font-family="Arial" font-size="11" text-anchor="middle">{value:.2f}</text>')
        parts.append(f'<text x="{x0 + 75}" y="{top + plot_h + 24}" font-family="Arial" font-size="13" text-anchor="middle">{html.escape(row.pool)}</text>')
    lx = width - 230
    legend = [("closed expected match", colors["closed"]), ("open-world disease agree", colors["open"]), ("open-world stable rate", colors["stable"])]
    for i, (label, color) in enumerate(legend):
        y = 74 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
    parts.extend(
        [
            '<text x="18" y="210" font-family="Arial" font-size="13" transform="rotate(-90 18 210)" text-anchor="middle">rate</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_confidence_curve(thresholds: pd.DataFrame, path: Path) -> None:
    width, height = 900, 410
    left, top, plot_w, plot_h = 74, 58, 690, 270
    colors = {"External-180": "#1565C0", "MultiSource-450": "#D9822B"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">High-confidence closed-set calls still often target mixed/unsupported open-world states</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    def xmap(v: float) -> float:
        return left + v / 0.9 * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - v * plot_h

    for pool, sub in thresholds.groupby("pool"):
        sub = sub.sort_values("confidence_threshold")
        pts = []
        for row in sub.itertuples(index=False):
            val = 0.0 if pd.isna(row.overcall_among_covered_vs_openworld) else float(row.overcall_among_covered_vs_openworld)
            pts.append(f"{xmap(float(row.confidence_threshold)):.1f},{ymap(val):.1f}")
            parts.append(f'<circle cx="{xmap(float(row.confidence_threshold)):.1f}" cy="{ymap(val):.1f}" r="4" fill="{colors.get(pool, "#555")}"/>')
        parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{colors.get(pool, "#555")}" stroke-width="3"/>')
    for xval in CONF_THRESHOLDS:
        parts.append(f'<text x="{xmap(xval):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{xval:.1f}</text>')
    lx = width - 170
    for i, (pool, color) in enumerate(colors.items()):
        y = 76 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{pool}</text>')
    parts.extend(
        [
            f'<text x="{left + plot_w / 2}" y="{height - 20}" font-family="Arial" font-size="13" text-anchor="middle">closed-set confidence threshold</text>',
            '<text x="18" y="215" font-family="Arial" font-size="13" transform="rotate(-90 18 215)" text-anchor="middle">overcall among covered samples</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    train_module = load_module(CODE_DIR / "train_rna_language_alignment.py", "predefined_label_train_module")
    taxonomy = load_module(CODE_DIR / "family_taxonomy.py", "closed_set_family_taxonomy")
    ckpt = torch_load(CHECKPOINT)

    x_train, train_meta = prepare_training_matrix(train_module, ckpt, split="train")
    x_test, test_meta = prepare_training_matrix(train_module, ckpt, split="test")
    y_train = make_training_labels(train_meta, taxonomy)
    y_test = make_training_labels(test_meta, taxonomy)
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
    y_pred_test = clf.predict(x_test)
    internal = {
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "n_classes": int(len(clf.classes_)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    all_predictions = []
    summary_rows = []
    threshold_rows = []
    transition_rows = []
    for pool, paths in POOLS.items():
        details = pd.read_csv(paths["details"])
        x_ext, ext_details = prepare_external_matrix(details, paths["inputs"], ckpt)
        pred, summary, thresholds = evaluate_pool(clf, x_ext, ext_details, pool, rng)
        all_predictions.append(pred)
        summary_rows.append(summary)
        threshold_rows.append(thresholds)
        transition = (
            pred.groupby(["closed_set_disease_family", "openworld_status"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["n", "closed_set_disease_family"], ascending=[False, True])
        )
        transition["pool"] = pool
        transition_rows.append(transition)

    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    thresholds_df = pd.concat(threshold_rows, ignore_index=True)
    transition_df = pd.concat(transition_rows, ignore_index=True)

    pred_df.to_csv(OUTDIR / "t2_closed_set_sample_predictions.csv", index=False)
    summary_df.to_csv(OUTDIR / "t2_pool_summary.csv", index=False)
    thresholds_df.to_csv(OUTDIR / "t2_confidence_thresholds.csv", index=False)
    transition_df.to_csv(OUTDIR / "t2_prediction_status_transition.csv", index=False)
    write_summary(summary_df, internal)
    svg_model_comparison(summary_df, OUTDIR / "t2_model_comparison.svg")
    svg_confidence_curve(thresholds_df, OUTDIR / "t2_confidence_overcall_curve.svg")


if __name__ == "__main__":
    main()
