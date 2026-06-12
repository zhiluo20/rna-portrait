#!/usr/bin/env python3
"""Export RNA-language alignment analysis outputs.

This script re-embeds the frozen semantic alignment backbone on its locked test
split and produces:
- paired RNA-text cosine vs shuffled RNA-text cosine controls,
- exact RNA-to-text and text-to-RNA retrieval lift over random,
- broad semantic retrieval for site, tumor status, and disease labels,
- bootstrap confidence intervals and SVG summaries.
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


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
OUTDIR = SUPP_DIR / "T1_RNA_language_alignment"

CHECKPOINT = BACKBONE_DIR / "bulk_multimodal_embedding.pt"
TEXT_EMBEDDINGS = BACKBONE_DIR / "caption_text_embeddings.npy"
SUMMARY_JSON = BACKBONE_DIR / "summary.json"
SAMPLE_EMBEDDINGS = BACKBONE_DIR / "sample_embeddings.csv"

BOOTSTRAPS = 1000
RNG_SEED = 20260604
TOPKS = (1, 5, 10)


def load_train_module():
    sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("rna_language_train_module", CODE_DIR / "train_rna_language_alignment.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load train module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int = BOOTSTRAPS) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = float(values[idx].mean())
    lo, hi = ci(boots)
    return float(values.mean()), lo, hi


def bootstrap_rate(indicator: np.ndarray, rng: np.random.Generator, n_boot: int = BOOTSTRAPS) -> Tuple[float, float, float]:
    return bootstrap_mean(np.asarray(indicator, dtype=float), rng, n_boot)


def normalize_label(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("unknown").str.strip().str.lower().replace({"": "unknown", "nan": "unknown", "none": "unknown"})


def random_any_match_rate(labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels)
    n = len(labels)
    if n <= 1:
        return 0.0
    counts = pd.Series(labels).value_counts().to_dict()
    probs = []
    for label in labels:
        same = counts.get(label, 0) - 1
        p = max(0.0, same / (n - 1))
        probs.append(1.0 - (1.0 - p) ** k)
    return float(np.mean(probs))


def exact_retrieval_metrics(sim: np.ndarray, rng: np.random.Generator) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    n = sim.shape[0]
    ranks = np.empty(n, dtype=int)
    exact_indicators: Dict[str, np.ndarray] = {}
    for i in range(n):
        order = np.argsort(-sim[i])
        ranks[i] = int(np.where(order == i)[0][0]) + 1
    rows = []
    for k in TOPKS:
        ind = (ranks <= k).astype(float)
        exact_indicators[f"r@{k}"] = ind
        mean, lo, hi = bootstrap_rate(ind, rng)
        random_rate = k / n
        rows.append(
            {
                "direction": "rna_to_text",
                "metric": f"r@{k}",
                "value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "random_baseline": random_rate,
                "lift_over_random": mean / random_rate if random_rate > 0 else np.nan,
            }
        )

    sim_t = sim.T
    ranks_t = np.empty(n, dtype=int)
    for i in range(n):
        order = np.argsort(-sim_t[i])
        ranks_t[i] = int(np.where(order == i)[0][0]) + 1
    for k in TOPKS:
        ind = (ranks_t <= k).astype(float)
        exact_indicators[f"text_to_rna_r@{k}"] = ind
        mean, lo, hi = bootstrap_rate(ind, rng)
        random_rate = k / n
        rows.append(
            {
                "direction": "text_to_rna",
                "metric": f"r@{k}",
                "value": mean,
                "ci_low": lo,
                "ci_high": hi,
                "random_baseline": random_rate,
                "lift_over_random": mean / random_rate if random_rate > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), exact_indicators


def broad_retrieval_metrics(sim: np.ndarray, meta: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    label_columns = {
        "site": "feat_anatomical_site",
        "tumor": "feat_tumor_status",
        "disease": "feat_disease_label",
    }
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    rows = []
    for label_name, col in label_columns.items():
        labels = normalize_label(meta[col]).to_numpy()
        valid = labels != "unknown"
        for k in TOPKS:
            ind = np.zeros(n, dtype=float)
            top = order[:, :k]
            for i in range(n):
                if not valid[i]:
                    ind[i] = np.nan
                    continue
                ind[i] = float(np.any(labels[top[i]] == labels[i]))
            valid_ind = ind[~np.isnan(ind)]
            mean, lo, hi = bootstrap_rate(valid_ind, rng)
            random_rate = random_any_match_rate(labels[valid], k)
            rows.append(
                {
                    "label": label_name,
                    "k": k,
                    "value": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "random_baseline": random_rate,
                    "lift_over_random": mean / random_rate if random_rate > 0 else np.nan,
                    "n_valid": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def cosine_control(expr_emb: np.ndarray, text_emb: np.ndarray, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paired = np.sum(expr_emb * text_emb, axis=1)
    n = len(paired)
    shuffled_values = []
    for _ in range(20):
        perm = rng.permutation(n)
        fixed = np.arange(n)
        bad = perm == fixed
        if bad.any():
            perm[bad] = np.roll(perm[bad], 1)
        shuffled_values.append(np.sum(expr_emb * text_emb[perm], axis=1))
    shuffled = np.concatenate(shuffled_values)

    control = pd.DataFrame(
        {
            "group": np.concatenate([np.repeat("paired", n), np.repeat("shuffled", len(shuffled))]),
            "cosine": np.concatenate([paired, shuffled]),
        }
    )
    rows = []
    for group, arr in [("paired", paired), ("shuffled", shuffled)]:
        mean, lo, hi = bootstrap_mean(arr, rng)
        rows.append({"group": group, "mean": mean, "ci_low": lo, "ci_high": hi, "n": int(len(arr))})
    diff_mean, diff_lo, diff_hi = bootstrap_mean(
        paired - shuffled.reshape(20, n).mean(axis=0),
        rng,
    )
    rows.append({"group": "paired_minus_shuffled", "mean": diff_mean, "ci_low": diff_lo, "ci_high": diff_hi, "n": int(n)})
    return control, pd.DataFrame(rows)


def svg_hist(control: pd.DataFrame, path: Path) -> None:
    paired = control.loc[control["group"] == "paired", "cosine"].to_numpy()
    shuffled = control.loc[control["group"] == "shuffled", "cosine"].to_numpy()
    lo = float(min(paired.min(), shuffled.min()))
    hi = float(max(paired.max(), shuffled.max()))
    bins = np.linspace(lo, hi, 46)
    p_counts, _ = np.histogram(paired, bins=bins, density=True)
    s_counts, _ = np.histogram(shuffled, bins=bins, density=True)
    max_y = max(float(p_counts.max()), float(s_counts.max()), 1e-9)
    width, height = 920, 430
    left, right, top, bottom = 70, 24, 52, 58
    plot_w, plot_h = width - left - right, height - top - bottom

    def xmap(v: float) -> float:
        return left + (v - lo) / (hi - lo) * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - v / max_y * plot_h

    def path_for(counts: np.ndarray) -> str:
        pts = []
        centers = (bins[:-1] + bins[1:]) / 2
        for x, y in zip(centers, counts):
            pts.append(f"{xmap(float(x)):.1f},{ymap(float(y)):.1f}")
        return "M " + " L ".join(pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="32" font-family="Arial" font-size="20" font-weight="700">Paired RNA-text cosine separates from shuffled controls</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<path d="{path_for(s_counts)}" fill="none" stroke="#6B7280" stroke-width="3"/>',
        f'<path d="{path_for(p_counts)}" fill="none" stroke="#1565C0" stroke-width="3"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 18}" font-family="Arial" font-size="13" text-anchor="middle">cosine similarity</text>',
        '<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">density</text>',
        f'<rect x="{width - 220}" y="70" width="14" height="14" fill="#1565C0"/>',
        f'<text x="{width - 200}" y="82" font-family="Arial" font-size="12">paired</text>',
        f'<rect x="{width - 220}" y="94" width="14" height="14" fill="#6B7280"/>',
        f'<text x="{width - 200}" y="106" font-family="Arial" font-size="12">shuffled</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_retrieval_bars(metrics: pd.DataFrame, path: Path) -> None:
    rows = metrics.loc[metrics["direction"] == "rna_to_text"].copy()
    width, height = 760, 390
    left, top = 80, 54
    bar_w, gap = 54, 56
    max_v = max(float(rows["value"].max()), float(rows["random_baseline"].max()), 1e-9)
    scale_max = max_v * 1.25
    plot_h = 260
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="32" font-family="Arial" font-size="20" font-weight="700">Exact RNA-to-text retrieval improves over random baseline</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 36}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    x = left + 52
    for row in rows.itertuples():
        value_h = row.value / scale_max * plot_h
        rand_h = row.random_baseline / scale_max * plot_h
        parts.append(f'<rect x="{x}" y="{top + plot_h - value_h:.1f}" width="{bar_w}" height="{value_h:.1f}" fill="#1565C0"/>')
        parts.append(f'<rect x="{x + bar_w + 8}" y="{top + plot_h - rand_h:.1f}" width="{bar_w}" height="{rand_h:.1f}" fill="#BDBDBD"/>')
        parts.append(f'<text x="{x + bar_w}" y="{top + plot_h + 20}" font-family="Arial" font-size="12" text-anchor="middle">{html.escape(row.metric)}</text>')
        parts.append(f'<text x="{x + bar_w}" y="{top + plot_h - value_h - 6:.1f}" font-family="Arial" font-size="11" text-anchor="middle">{row.lift_over_random:.1f}x</text>')
        x += bar_w * 2 + gap
    parts.extend(
        [
            f'<rect x="{width - 210}" y="70" width="14" height="14" fill="#1565C0"/>',
            f'<text x="{width - 190}" y="82" font-family="Arial" font-size="12">observed</text>',
            f'<rect x="{width - 210}" y="94" width="14" height="14" fill="#BDBDBD"/>',
            f'<text x="{width - 190}" y="106" font-family="Arial" font-size="12">random</text>',
            '<text x="18" y="210" font-family="Arial" font-size="13" transform="rotate(-90 18 210)" text-anchor="middle">retrieval rate</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_broad_recall(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics.loc[metrics["k"].isin([1, 5, 10])].copy()
    width, height = 980, 440
    left, top = 72, 58
    plot_h = 280
    labels = ["site", "tumor", "disease"]
    colors = {"site": "#1565C0", "tumor": "#2E7D32", "disease": "#D9822B"}
    y_max = min(1.0, max(0.1, float(sub["value"].max()) * 1.18))
    group_w = 250
    bar_w = 22
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Broad semantic retrieval remains strongest for site and tumor context</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - 40}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for gi, k in enumerate(TOPKS):
        x0 = left + 70 + gi * group_w
        for li, label in enumerate(labels):
            row = sub[(sub["k"] == k) & (sub["label"] == label)].iloc[0]
            h = float(row.value) / y_max * plot_h
            x = x0 + li * (bar_w + 18)
            y = top + plot_h - h
            parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{colors[label]}"/>')
            ci_y1 = top + plot_h - float(row.ci_low) / y_max * plot_h
            ci_y2 = top + plot_h - float(row.ci_high) / y_max * plot_h
            cx = x + bar_w / 2
            parts.append(f'<line x1="{cx:.1f}" y1="{ci_y1:.1f}" x2="{cx:.1f}" y2="{ci_y2:.1f}" stroke="#111"/>')
        parts.append(f'<text x="{x0 + 48}" y="{top + plot_h + 22}" font-family="Arial" font-size="13" text-anchor="middle">top-{k}</text>')
    lx = width - 230
    for i, label in enumerate(labels):
        y = 74 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{colors[label]}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{label}</text>')
    parts.extend(
        [
            '<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">any top-k semantic match</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(cosine_summary: pd.DataFrame, exact: pd.DataFrame, broad: pd.DataFrame, meta: pd.DataFrame) -> None:
    paired = cosine_summary.loc[cosine_summary["group"] == "paired"].iloc[0]
    shuffled = cosine_summary.loc[cosine_summary["group"] == "shuffled"].iloc[0]
    diff = cosine_summary.loc[cosine_summary["group"] == "paired_minus_shuffled"].iloc[0]
    rna_r10 = exact[(exact["direction"] == "rna_to_text") & (exact["metric"] == "r@10")].iloc[0]
    site_top1 = broad[(broad["label"] == "site") & (broad["k"] == 1)].iloc[0]
    tumor_top1 = broad[(broad["label"] == "tumor") & (broad["k"] == 1)].iloc[0]
    disease_top1 = broad[(broad["label"] == "disease") & (broad["k"] == 1)].iloc[0]
    lines = [
        "# RNA-language alignment analysis",
        "",
        "## Purpose",
        "",
        "This analysis re-embeds the frozen semantic alignment backbone on its locked test split and evaluates whether paired RNA-text representations are aligned beyond shuffled/random controls.",
        "",
        "## Data",
        "",
        f"- Test samples: `{len(meta)}`",
        f"- Unique projects/GSE: `{meta['gse'].nunique()}`",
        f"- Source datasets: `{meta.get('source_dataset', pd.Series(dtype=str)).value_counts().to_dict()}`",
        "",
        "## Main findings",
        "",
        f"- Paired RNA-text cosine mean: `{paired['mean']:.4f}` (95% CI `{paired['ci_low']:.4f}`-`{paired['ci_high']:.4f}`)",
        f"- Shuffled RNA-text cosine mean: `{shuffled['mean']:.4f}` (95% CI `{shuffled['ci_low']:.4f}`-`{shuffled['ci_high']:.4f}`)",
        f"- Paired-minus-shuffled mean difference: `{diff['mean']:.4f}` (95% CI `{diff['ci_low']:.4f}`-`{diff['ci_high']:.4f}`)",
        f"- Exact RNA-to-text R@10: `{rna_r10['value']:.4f}`, random baseline `{rna_r10['random_baseline']:.4f}`, lift `{rna_r10['lift_over_random']:.1f}x`",
        f"- Broad top-1 site match: `{site_top1['value']:.4f}` vs random `{site_top1['random_baseline']:.4f}`",
        f"- Broad top-1 tumor-status match: `{tumor_top1['value']:.4f}` vs random `{tumor_top1['random_baseline']:.4f}`",
        f"- Broad top-1 disease-label match: `{disease_top1['value']:.4f}` vs random `{disease_top1['random_baseline']:.4f}`",
        "",
        "## Output files",
        "",
        "- `t1_cosine_controls.csv`: paired and shuffled cosine values.",
        "- `t1_cosine_summary.csv`: mean cosine and bootstrap confidence intervals.",
        "- `t1_exact_retrieval_metrics.csv`: exact caption retrieval metrics and random baseline lift.",
        "- `t1_broad_semantic_retrieval.csv`: site/tumor/disease top-k semantic match metrics.",
        "- `t1_paired_vs_shuffled_cosine.svg`: cosine distribution control.",
        "- `t1_exact_retrieval_lift.svg`: exact retrieval versus random baseline.",
        "- `t1_broad_semantic_recall.svg`: broad semantic recall by label type.",
        "",
        "## Interpretation",
        "",
        "Matched RNA profiles and natural-language metadata show higher similarity than shuffled controls. Exact paired-text retrieval remains difficult, so these outputs should be interpreted together with broader semantic retrieval metrics and downstream portrait analyses.",
        "",
    ]
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    module = load_train_module()
    ckpt = torch_load(CHECKPOINT)
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    bundle = module.load_dataset()
    meta = bundle.meta.copy()
    meta["caption_text"] = meta.apply(module.build_caption, axis=1)
    if "split_assignments" in ckpt:
        split_assignments = {str(k): str(v) for k, v in ckpt["split_assignments"].items()}
    else:
        saved_samples = pd.read_csv(SAMPLE_EMBEDDINGS, usecols=["sample_id", "split"])
        split_assignments = dict(zip(saved_samples["sample_id"].astype(str), saved_samples["split"].astype(str)))
    meta["split"] = meta["sample_id"].astype(str).map(split_assignments)
    test_meta = meta.loc[meta["split"] == "test"].copy().reset_index(drop=True)
    if len(test_meta) != summary["split_counts"]["test"]:
        raise RuntimeError(f"test split size mismatch: {len(test_meta)} vs {summary['split_counts']['test']}")

    selected_genes = [str(g) for g in ckpt["selected_genes"]]
    text_matrix = np.load(TEXT_EMBEDDINGS).astype(np.float32)
    text_meta = meta[["sample_id", "caption_text"]].drop_duplicates("sample_id").reset_index(drop=True)
    text_row_index = {str(sample_id): idx for idx, sample_id in enumerate(text_meta["sample_id"].tolist())}

    expr = module.load_expr_subset(bundle, selected_genes, test_meta["sample_id"].astype(str).tolist())
    split = module.prepare_split_data(
        expr,
        test_meta,
        selected_genes,
        text_matrix,
        text_row_index,
        np.asarray(ckpt["expr_mean"], dtype=np.float32),
        np.asarray(ckpt["expr_std"], dtype=np.float32),
        float(ckpt["age_mean"]),
        float(ckpt["age_std"]),
        {str(k): int(v) for k, v in ckpt["source_map"].items()},
    )

    device = torch.device("cpu")
    model = module.BulkRNALanguageAligner(
        n_genes=len(selected_genes),
        n_text_features=int(ckpt["config"]["n_text_features"]),
        n_sources=len(ckpt["source_map"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    expr_emb, text_emb, _, _ = module.embed_split(model, split, device)
    sim = expr_emb @ text_emb.T

    control, cosine_summary = cosine_control(expr_emb, text_emb, rng)
    exact_metrics, _ = exact_retrieval_metrics(sim, rng)
    broad_metrics = broad_retrieval_metrics(sim, test_meta, rng)

    control.to_csv(OUTDIR / "t1_cosine_controls.csv", index=False)
    cosine_summary.to_csv(OUTDIR / "t1_cosine_summary.csv", index=False)
    exact_metrics.to_csv(OUTDIR / "t1_exact_retrieval_metrics.csv", index=False)
    broad_metrics.to_csv(OUTDIR / "t1_broad_semantic_retrieval.csv", index=False)
    test_meta[["sample_id", "gse", "feat_anatomical_site", "feat_tumor_status", "feat_disease_label", "split"]].to_csv(
        OUTDIR / "t1_test_samples.csv", index=False
    )

    svg_hist(control, OUTDIR / "t1_paired_vs_shuffled_cosine.svg")
    svg_retrieval_bars(exact_metrics, OUTDIR / "t1_exact_retrieval_lift.svg")
    svg_broad_recall(broad_metrics, OUTDIR / "t1_broad_semantic_recall.svg")
    write_summary(cosine_summary, exact_metrics, broad_metrics, test_meta)


if __name__ == "__main__":
    main()
