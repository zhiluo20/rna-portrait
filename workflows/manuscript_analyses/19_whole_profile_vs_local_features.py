#!/usr/bin/env python3
"""T12 whole-profile RNA portraits versus local-part baselines.

This analysis asks whether the RNA portrait is merely a restatement of labels
or local marker/pathway/deconvolution features. It uses grouped cross-validation
by project so that project-specific shortcuts are not rewarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
OUTDIR = SUPP_DIR / "T12_whole_profile_vs_local_parts"
INPUT = SUPP_DIR / "T11_failure_mode_reliability_boundaries" / "t11_sample_reliability_table.csv"

RNG_SEED = 20260605
N_PERMUTATIONS = 80
N_SPLITS = 5
PORTRAIT_COL = "semantic_state_family"
GROUP_COL = "project"

LABEL_COLS = [
    "pool",
    "project",
    "expected_site_family",
    "expected_disease_family",
    "closed_set_disease_family",
]

LOCAL_PART_COLS = [
    "immune_core_z",
    "t_cell_nk_z",
    "myeloid_inflammation_z",
    "hematologic_lineage_z",
    "epithelial_z",
    "stromal_ecm_z",
    "proliferation_z",
    "interferon_z",
    "ifn_alpha_score",
    "ifn_gamma_score",
    "tnfa_nfkb_score",
    "inflammatory_response_score",
    "complement_score",
    "t_cell_cytotoxic_score",
    "myeloid_activation_score",
    "emt_stromal_score",
    "epithelial_identity_score",
    "g2m_checkpoint_score",
    "e2f_targets_score",
    "hypoxia_score",
    "angiogenesis_score",
    "epic_immune_fraction_z",
    "epic_tcell_fraction_z",
    "epic_macrophage_fraction_z",
    "epic_caf_fraction_z",
    "epic_endothelial_fraction_z",
    "epic_stromal_fraction_z",
    "epic_other_fraction_z",
    "mcp_immune_z_mean",
    "mcp_t_nk_cytotoxic_z_mean",
    "mcp_myeloid_z_mean",
    "mcp_stromal_z_mean",
    "mcp_endothelial_z",
    "mcp_fibroblast_z",
]

EVIDENCE_OUTCOMES = [
    "immune_support",
    "context_support",
    "tumor_like_support",
    "mixed_evidence_support",
    "clean_context_support",
]

HIGHER_LEVEL_OUTCOMES = [
    "partial_or_strong_claim_rate",
    "boundary_flag_count",
]

OUTCOME_LABELS = {
    "immune_support": "immune evidence",
    "context_support": "context evidence",
    "tumor_like_support": "tumor-like evidence",
    "mixed_evidence_support": "multi-signal evidence",
    "clean_context_support": "clean-context evidence",
    "partial_or_strong_claim_rate": "claim support rate",
    "boundary_flag_count": "boundary flag count",
}

PREDICTOR_SETS = {
    "labels_only": ("Labels only", LABEL_COLS, []),
    "local_parts_only": ("Local parts only", [], LOCAL_PART_COLS),
    "labels_plus_local_parts": ("Labels + local parts", LABEL_COLS, LOCAL_PART_COLS),
    "labels_local_plus_portrait": ("Labels + local parts + portrait", [*LABEL_COLS, PORTRAIT_COL], LOCAL_PART_COLS),
}

REDUCIBILITY_PREDICTOR_SETS = {
    "majority_baseline": "Majority baseline",
    "labels_only": "Labels only",
    "local_parts_only": "Local parts only",
    "labels_plus_local_parts": "Labels + local parts",
}

PORTRAIT_LABELS = {
    "hematologic_override": "hematologic",
    "epithelial_override": "epithelial",
    "stable_consensus": "stable",
    "clean_anchor_override": "clean anchor",
    "generic_context_override": "context",
    "unsupported_semantics": "unsupported",
    "family_conflict": "conflict",
    "other": "other",
}


@dataclass
class RegressionResult:
    outcome: str
    predictor_set: str
    predictor_label: str
    n_samples: int
    cv_r2: float


def save_figure(fig: plt.Figure, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTDIR / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def clip_text(value: object, width: int) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def existing(cols: Sequence[str], df: pd.DataFrame) -> List[str]:
    return [col for col in cols if col in df.columns]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df = df.copy()
    df[PORTRAIT_COL] = df[PORTRAIT_COL].fillna("missing").astype(str)
    for col in existing([*LOCAL_PART_COLS, *EVIDENCE_OUTCOMES, *HIGHER_LEVEL_OUTCOMES], df):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in existing(LABEL_COLS, df):
        df[col] = df[col].fillna("missing").astype(str)
    return df


def split_indices(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = df[GROUP_COL].fillna("missing").astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(N_SPLITS, n_groups)
    if n_splits < 2:
        raise ValueError("Need at least two project groups for grouped cross-validation.")
    return list(GroupKFold(n_splits=n_splits).split(df, groups=groups))


def preprocessing(cat_cols: Sequence[str], num_cols: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)))
    if num_cols:
        transformers.append(
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(num_cols),
            )
        )
    return ColumnTransformer(transformers)


def regression_oof(
    df: pd.DataFrame,
    outcome: str,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [*cat_cols, *num_cols]
    y_raw = pd.to_numeric(df[outcome], errors="coerce").to_numpy()
    ok = np.isfinite(y_raw)
    y = y_raw[ok]
    x = df.loc[ok, feature_cols].copy()
    ok_index = np.where(ok)[0]
    preds = np.full(len(y), np.nan)
    mapped_splits = []
    ok_lookup = {idx: i for i, idx in enumerate(ok_index)}
    for train_idx, test_idx in splits:
        train = np.array([ok_lookup[i] for i in train_idx if i in ok_lookup], dtype=int)
        test = np.array([ok_lookup[i] for i in test_idx if i in ok_lookup], dtype=int)
        if len(train) == 0 or len(test) == 0:
            continue
        mapped_splits.append((train, test))
        model = make_pipeline(preprocessing(cat_cols, num_cols), Ridge(alpha=10.0))
        model.fit(x.iloc[train], y[train])
        preds[test] = model.predict(x.iloc[test])
    valid = np.isfinite(preds)
    return y[valid], preds[valid], ok_index[valid]


def regression_comparison(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    local_cols = existing(LOCAL_PART_COLS, df)
    label_cols = existing(LABEL_COLS, df)
    outcomes = [col for col in [*EVIDENCE_OUTCOMES, *HIGHER_LEVEL_OUTCOMES] if col in df.columns]
    splits = split_indices(df)
    rows: List[RegressionResult] = []
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    resolved_predictors = {
        "labels_only": ("Labels only", label_cols, []),
        "local_parts_only": ("Local parts only", [], local_cols),
        "labels_plus_local_parts": ("Labels + local parts", label_cols, local_cols),
        "labels_local_plus_portrait": ("Labels + local parts + portrait", [*label_cols, PORTRAIT_COL], local_cols),
    }

    for outcome in outcomes:
        for predictor_id, (predictor_label, cat_cols, num_cols) in resolved_predictors.items():
            y, pred, row_idx = regression_oof(df, outcome, cat_cols, num_cols, splits)
            r2 = float(r2_score(y, pred)) if len(y) else np.nan
            rows.append(
                RegressionResult(
                    outcome=outcome,
                    predictor_set=predictor_id,
                    predictor_label=predictor_label,
                    n_samples=len(y),
                    cv_r2=r2,
                )
            )
            predictions[(outcome, predictor_id)] = (y, pred, row_idx)

    comparison = pd.DataFrame([row.__dict__ for row in rows])
    increment_rows = []
    rng = np.random.default_rng(RNG_SEED)
    for outcome in outcomes:
        base = comparison[(comparison["outcome"] == outcome) & (comparison["predictor_set"] == "labels_plus_local_parts")].iloc[0]
        full = comparison[(comparison["outcome"] == outcome) & (comparison["predictor_set"] == "labels_local_plus_portrait")].iloc[0]
        observed_delta = float(full["cv_r2"] - base["cv_r2"])
        permuted = []
        for _ in range(N_PERMUTATIONS):
            shuffled = df.copy()
            shuffled["portrait_permuted_for_t12"] = rng.permutation(shuffled[PORTRAIT_COL].to_numpy())
            y, pred, _ = regression_oof(
                shuffled,
                outcome,
                [*label_cols, "portrait_permuted_for_t12"],
                local_cols,
                splits,
            )
            perm_r2 = float(r2_score(y, pred)) if len(y) else np.nan
            permuted.append(perm_r2 - float(base["cv_r2"]))
        perm_arr = np.asarray(permuted, dtype=float)
        increment_rows.append(
            {
                "outcome": outcome,
                "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                "n_samples": int(full["n_samples"]),
                "base_cv_r2_labels_plus_local": float(base["cv_r2"]),
                "full_cv_r2_with_portrait": float(full["cv_r2"]),
                "portrait_incremental_cv_r2": observed_delta,
                "permuted_incremental_mean": float(np.nanmean(perm_arr)),
                "permuted_incremental_q95": float(np.nanquantile(perm_arr, 0.95)),
                "empirical_p_greater": float((1 + np.sum(perm_arr >= observed_delta)) / (len(perm_arr) + 1)),
                "passes_perm_q95": bool(observed_delta > np.nanquantile(perm_arr, 0.95)),
                "outcome_family": "aggregate evidence axis" if outcome in EVIDENCE_OUTCOMES else "higher-level portrait outcome",
            }
        )
    return comparison, pd.DataFrame(increment_rows), predictions


def classification_oof(
    df: pd.DataFrame,
    target: str,
    cat_cols: Sequence[str],
    num_cols: Sequence[str],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    dummy: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = [*cat_cols, *num_cols]
    x = df[feature_cols].copy() if feature_cols else pd.DataFrame(index=df.index)
    y = df[target].astype(str).to_numpy()
    preds = np.empty(len(y), dtype=object)
    for train, test in splits:
        if dummy:
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = make_pipeline(
                preprocessing(cat_cols, num_cols),
                LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs"),
            )
        model.fit(x.iloc[train], y[train])
        preds[test] = model.predict(x.iloc[test])
    return y, preds


def portrait_reducibility(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = existing(LABEL_COLS, df)
    local_cols = existing(LOCAL_PART_COLS, df)
    counts = df[PORTRAIT_COL].value_counts()
    keep = counts[counts >= 10].index
    sub = df[df[PORTRAIT_COL].isin(keep)].copy().reset_index(drop=True)
    splits = split_indices(sub)
    specs = {
        "majority_baseline": ("Majority baseline", [], [], True),
        "labels_only": ("Labels only", label_cols, [], False),
        "local_parts_only": ("Local parts only", [], local_cols, False),
        "labels_plus_local_parts": ("Labels + local parts", label_cols, local_cols, False),
    }
    rows = []
    for predictor_id, (label, cat_cols, num_cols, dummy) in specs.items():
        y, pred = classification_oof(sub, PORTRAIT_COL, cat_cols, num_cols, splits, dummy=dummy)
        rows.append(
            {
                "predictor_set": predictor_id,
                "predictor_label": label,
                "n_samples": len(y),
                "n_portrait_classes": len(np.unique(y)),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
            }
        )
    return pd.DataFrame(rows)


def representative_improvement_cases(
    df: pd.DataFrame,
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    case_outcomes = ["boundary_flag_count", "partial_or_strong_claim_rate"]
    for outcome in case_outcomes:
        if (outcome, "labels_plus_local_parts") not in predictions or (outcome, "labels_local_plus_portrait") not in predictions:
            continue
        y_base, pred_base, idx = predictions[(outcome, "labels_plus_local_parts")]
        y_full, pred_full, idx_full = predictions[(outcome, "labels_local_plus_portrait")]
        if not np.array_equal(idx, idx_full):
            continue
        improvement = np.abs(y_base - pred_base) - np.abs(y_full - pred_full)
        order = np.argsort(improvement)[::-1]
        for rank in order[:8]:
            row = df.iloc[int(idx[rank])]
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": OUTCOME_LABELS.get(outcome, outcome),
                    "sample_id": row.get("sample_id", row.get("file", "")),
                    "pool": row.get("pool", ""),
                    "project": row.get("project", ""),
                    "expected_site_family": row.get("expected_site_family", ""),
                    "expected_disease_family": row.get("expected_disease_family", ""),
                    "semantic_state_family": row.get(PORTRAIT_COL, ""),
                    "portrait_label": PORTRAIT_LABELS.get(str(row.get(PORTRAIT_COL, "")), str(row.get(PORTRAIT_COL, ""))),
                    "observed": float(y_base[rank]),
                    "label_local_prediction": float(pred_base[rank]),
                    "with_portrait_prediction": float(pred_full[rank]),
                    "absolute_error_reduction": float(improvement[rank]),
                    "boundary_flags": row.get("boundary_flags", ""),
                    "reliability_tier": row.get("reliability_tier", ""),
                    "portrait_text_for_grounding": row.get("portrait_text_for_grounding", ""),
                }
            )
    return pd.DataFrame(rows)


def draw_prediction_heatmap(comparison: pd.DataFrame) -> None:
    order_cols = ["labels_only", "local_parts_only", "labels_plus_local_parts", "labels_local_plus_portrait"]
    order_rows = [*EVIDENCE_OUTCOMES, *HIGHER_LEVEL_OUTCOMES]
    plot = comparison.pivot_table(index="outcome", columns="predictor_set", values="cv_r2")
    plot = plot.reindex(index=[row for row in order_rows if row in plot.index], columns=order_cols)
    labels_x = ["labels", "local parts", "labels + local", "+ portrait"]
    labels_y = [OUTCOME_LABELS.get(idx, idx) for idx in plot.index]
    data = plot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=max(1.0, np.nanmax(data)))
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#111827")
    ax.set_title("What can labels, local parts and RNA portraits predict?", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("grouped cross-validated R2")
    save_figure(fig, "t12_prediction_comparison_heatmap")


def draw_incremental_value(increment: pd.DataFrame) -> None:
    plot = increment.copy()
    plot["order"] = plot["outcome"].map({outcome: i for i, outcome in enumerate([*EVIDENCE_OUTCOMES, *HIGHER_LEVEL_OUTCOMES])})
    plot = plot.sort_values("order")
    y = np.arange(len(plot))
    colors = ["#64748B" if fam == "aggregate evidence axis" else "#1B9E77" for fam in plot["outcome_family"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axvline(0, color="#94A3B8", lw=1)
    ax.barh(y, plot["portrait_incremental_cv_r2"], color=colors, height=0.55, zorder=2)
    ax.scatter(plot["permuted_incremental_q95"], y, color="#B91C1C", s=26, label="shuffled portrait 95% line", zorder=3)
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(row.portrait_incremental_cv_r2 + 0.006, i, f"{row.portrait_incremental_cv_r2:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["outcome_label"])
    ax.set_xlabel("incremental cross-validated R2 after labels + local parts")
    ax.set_title("Portrait adds little to aggregate local axes, but helps higher-level boundaries", pad=8)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    save_figure(fig, "t12_incremental_value_after_local_parts")


def draw_reducibility(reducibility: pd.DataFrame) -> None:
    plot = reducibility.copy()
    order = ["majority_baseline", "labels_only", "local_parts_only", "labels_plus_local_parts"]
    plot["order"] = plot["predictor_set"].map({key: i for i, key in enumerate(order)})
    plot = plot.sort_values("order")
    x = np.arange(len(plot))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    ax.bar(x - width, plot["accuracy"], width=width, color="#94A3B8", label="accuracy")
    ax.bar(x, plot["balanced_accuracy"], width=width, color="#3B6FB6", label="balanced accuracy")
    ax.bar(x + width, plot["macro_f1"], width=width, color="#1B9E77", label="macro F1")
    for i, row in enumerate(plot.itertuples(index=False)):
        ax.text(i, max(row.accuracy, row.balanced_accuracy, row.macro_f1) + 0.025, f"n={row.n_samples}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["predictor_label"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("grouped cross-validated score")
    ax.set_title("Can labels or local parts fully recover the RNA portrait group?", pad=8)
    ax.legend(frameon=False, ncols=3, loc="upper right", fontsize=8)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    save_figure(fig, "t12_portrait_reducibility_bar")


def draw_cases(cases: pd.DataFrame) -> None:
    if cases.empty:
        return
    show = cases.sort_values("absolute_error_reduction", ascending=False).head(14)
    fig_h = max(5.6, 0.56 * len(show) + 1.0)
    fig, ax = plt.subplots(figsize=(12.0, fig_h))
    ax.axis("off")
    ax.text(
        0.01,
        0.98,
        "Representative cases where the portrait improves higher-level prediction",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    y0 = 0.91
    dy = 0.84 / max(len(show), 1)
    for i, row in enumerate(show.itertuples(index=False)):
        y = y0 - i * dy
        color = "#EAF3FF" if row.outcome == "boundary_flag_count" else "#F0FDF4"
        ax.add_patch(
            plt.Rectangle(
                (0.01, y - dy * 0.78),
                0.98,
                dy * 0.70,
                facecolor=color,
                edgecolor="#E5E7EB",
                transform=ax.transAxes,
            )
        )
        title = clip_text(f"{row.outcome_label} | {row.portrait_label} | {row.reliability_tier}", 44)
        detail = (
            f"obs {row.observed:.2f}\n"
            f"label+local {row.label_local_prediction:.2f} -> +portrait {row.with_portrait_prediction:.2f}\n"
            f"error drop {row.absolute_error_reduction:.2f}"
        )
        flags = clip_text(str(row.boundary_flags).replace("|", ", "), 70)
        sample = clip_text(str(row.sample_id), 48)
        ax.text(0.025, y - dy * 0.30, title, fontsize=7.8, fontweight="bold", color="#244F86", transform=ax.transAxes)
        ax.text(0.025, y - dy * 0.58, sample, fontsize=6.9, color="#64748B", transform=ax.transAxes)
        ax.text(0.40, y - dy * 0.30, detail, fontsize=7.2, color="#111827", transform=ax.transAxes, linespacing=1.25)
        ax.text(0.69, y - dy * 0.38, flags, fontsize=7.1, color="#374151", transform=ax.transAxes)
    save_figure(fig, "t12_representative_improvement_cases")


def write_summary(comparison: pd.DataFrame, increment: pd.DataFrame, reducibility: pd.DataFrame, cases: pd.DataFrame) -> None:
    evidence_inc = increment[increment["outcome_family"] == "aggregate evidence axis"]
    high_inc = increment[increment["outcome_family"] == "higher-level portrait outcome"]
    mean_local = comparison[
        (comparison["predictor_set"] == "local_parts_only") & (comparison["outcome"].isin(EVIDENCE_OUTCOMES))
    ]["cv_r2"].mean()
    mean_evidence_delta = evidence_inc["portrait_incremental_cv_r2"].mean()
    claim_row = increment[increment["outcome"] == "partial_or_strong_claim_rate"].iloc[0] if "partial_or_strong_claim_rate" in set(increment["outcome"]) else None
    boundary_row = increment[increment["outcome"] == "boundary_flag_count"].iloc[0] if "boundary_flag_count" in set(increment["outcome"]) else None
    label_local = reducibility[reducibility["predictor_set"] == "labels_plus_local_parts"].iloc[0]
    lines = [
        "# Whole-profile portraits versus local-part baselines",
        "",
        "## What this analysis asks",
        "",
        "This analysis asks whether the RNA portrait is reducible to labels, marker scores, pathway scores and deconvolution scores. It uses grouped cross-validation by project, so the model is tested on held-out projects.",
        "",
        "The analysis tests whether portrait outputs are reducible to labels and local evidence summaries.",
        "",
        "## Headline result",
        "",
        f"- Local parts nearly reconstruct the aggregate evidence axes by construction: mean local-part CV R2 across five evidence axes is `{mean_local:.3f}`.",
        f"- Adding the RNA portrait after labels + local parts gives mean incremental CV R2 `{mean_evidence_delta:.3f}` on those aggregate evidence axes.",
    ]
    if claim_row is not None:
        lines.append(
            f"- For claim support rate, adding the portrait gives incremental CV R2 `{claim_row.portrait_incremental_cv_r2:.3f}` after labels + local parts."
        )
    if boundary_row is not None:
        lines.append(
            f"- For reliability boundary count, adding the portrait gives incremental CV R2 `{boundary_row.portrait_incremental_cv_r2:.3f}` after labels + local parts."
        )
    lines.extend(
        [
            f"- Labels + local parts do not fully recover the portrait group: accuracy `{label_local.accuracy:.3f}`, balanced accuracy `{label_local.balanced_accuracy:.3f}`, macro F1 `{label_local.macro_f1:.3f}` across major portrait classes.",
            "",
            "## Interpretation",
            "",
            "Aggregate biology axes are local-part summaries by construction, but the RNA portrait is not fully reducible to disease/site/source labels or those local summaries. The strongest added value appears in higher-level claim grounding and reliability-boundary organization, not in re-predicting the same aggregate evidence axes.",
            "",
            "This should be interpreted as a reducibility analysis rather than as a claim that portraits replace marker, pathway or deconvolution evidence.",
            "",
            "## Output files",
            "",
            "- `t12_cv_model_comparison.csv`: grouped-CV R2 for labels, local parts and portrait-augmented models.",
            "- `t12_incremental_value_after_local_parts.csv`: portrait incremental value after labels + local parts, with shuffled-portrait baselines.",
            "- `t12_portrait_reducibility.csv`: grouped-CV classification test asking whether labels/local parts can recover portrait groups.",
            "- `t12_representative_improvement_cases.csv`: samples where adding the portrait most improves higher-level outcome prediction.",
            "- `t12_prediction_comparison_heatmap.svg/png`: predictor comparison heatmap.",
            "- `t12_incremental_value_after_local_parts.svg/png`: incremental portrait value plot.",
            "- `t12_portrait_reducibility_bar.svg/png`: portrait reducibility plot.",
            "- `t12_representative_improvement_cases.svg/png`: representative improvement cases.",
            "",
            "Marker, pathway and deconvolution evidence remain important audit axes for the portrait readout.",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    comparison, increment, predictions = regression_comparison(df)
    reducibility = portrait_reducibility(df)
    cases = representative_improvement_cases(df, predictions)

    comparison.to_csv(OUTDIR / "t12_cv_model_comparison.csv", index=False)
    increment.to_csv(OUTDIR / "t12_incremental_value_after_local_parts.csv", index=False)
    reducibility.to_csv(OUTDIR / "t12_portrait_reducibility.csv", index=False)
    cases.to_csv(OUTDIR / "t12_representative_improvement_cases.csv", index=False)

    draw_prediction_heatmap(comparison)
    draw_incremental_value(increment)
    draw_reducibility(reducibility)
    draw_cases(cases)
    write_summary(comparison, increment, reducibility, cases)
    print(OUTDIR)


if __name__ == "__main__":
    main()
