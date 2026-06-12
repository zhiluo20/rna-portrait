#!/usr/bin/env python3
"""Calibrated predefined disease-family baseline."""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


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
OUTDIR = SUPP_DIR / "T2_calibrated_closed_set"
T2_SCRIPT = SCRIPT_DIR / "02_predefined_label_baseline.py"

RNG_SEED = 20260604
MIN_TRAIN_CLASS_N = 10
CONFIDENCE_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]
RISK_COVERAGES = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1]
RELIABILITY_BINS = np.linspace(0.0, 1.0, 11)


def load_module(path: Path, module_name: str):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_labels_with_train_vocab(meta: pd.DataFrame, taxonomy, keep: set[str] | None = None) -> Tuple[pd.Series, set[str]]:
    raw = meta.apply(
        lambda r: taxonomy.disease_family(r.get("feat_disease_label"), r.get("feat_anatomical_site"), r.get("feat_tumor_status")),
        axis=1,
    ).astype(str)
    if keep is None:
        counts = raw.value_counts()
        keep = set(counts[counts >= MIN_TRAIN_CLASS_N].index)
    mapped = raw.where(raw.isin(keep), "rare_or_unmapped")
    return mapped.astype(str), keep


def map_expected_family(series: pd.Series, keep: set[str]) -> pd.Series:
    raw = series.astype(str)
    return raw.where(raw.isin(keep), "rare_or_unmapped")


def train_base_classifier(x_train: np.ndarray, y_train: pd.Series) -> SGDClassifier:
    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=80,
        tol=1e-3,
        class_weight="balanced",
        random_state=RNG_SEED,
        n_jobs=1,
    )
    clf.fit(x_train, y_train)
    return clf


def calibrate_prefit(base: SGDClassifier, x_val: np.ndarray, y_val: pd.Series, method: str) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(estimator=base, method=method, cv="prefit", ensemble=False)
    cal.fit(x_val, y_val)
    return cal


def aligned_decision_scores(model: SGDClassifier, x: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    scores = np.asarray(model.decision_function(x), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    model_classes = np.asarray(model.classes_).astype(str)
    aligned = np.full((x.shape[0], len(class_order)), -1e6, dtype=np.float64)
    class_to_col = {str(c): i for i, c in enumerate(class_order)}
    for j, cls in enumerate(model_classes):
        if str(cls) in class_to_col:
            aligned[:, class_to_col[str(cls)]] = scores[:, j]
    return np.nan_to_num(aligned, nan=-1e6, posinf=1e6, neginf=-1e6)


def softmax_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    scaled = scores / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs


def stable_decision_softmax(model: SGDClassifier, x: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    scores = aligned_decision_scores(model, x, class_order)
    return softmax_scores(scores, temperature=1.0)


class TemperatureScaledClassifier:
    def __init__(self, base: SGDClassifier, temperature: float, class_order: np.ndarray):
        self.base = base
        self.temperature = float(temperature)
        self.classes_ = np.asarray(base.classes_).astype(str)
        self.class_order = np.asarray(class_order).astype(str)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = aligned_decision_scores(self.base, x, self.class_order)
        return softmax_scores(scores, temperature=self.temperature)


def fit_temperature_scaled(base: SGDClassifier, x_val: np.ndarray, y_val: pd.Series, class_order: np.ndarray) -> TemperatureScaledClassifier:
    scores = aligned_decision_scores(base, x_val, class_order)
    class_to_col = {str(c): i for i, c in enumerate(class_order)}
    y_idx = np.asarray([class_to_col.get(str(y), -1) for y in y_val.astype(str)], dtype=int)
    valid = y_idx >= 0
    scores = scores[valid]
    y_idx = y_idx[valid]
    candidates = np.logspace(-1, 3, 220)
    best_temp = 1.0
    best_nll = float("inf")
    for temp in candidates:
        probs = softmax_scores(scores, temperature=float(temp))
        p_true = np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
        nll = float(-np.log(p_true).mean())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return TemperatureScaledClassifier(base, best_temp, class_order)


def aligned_predict_proba(model, x: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    if isinstance(model, SGDClassifier):
        return stable_decision_softmax(model, x, class_order)
    if isinstance(model, TemperatureScaledClassifier):
        return model.predict_proba(x)
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    model_classes = np.asarray(model.classes_).astype(str)
    out = np.zeros((x.shape[0], len(class_order)), dtype=np.float64)
    class_to_col = {str(c): i for i, c in enumerate(class_order)}
    for j, cls in enumerate(model_classes):
        if str(cls) in class_to_col:
            out[:, class_to_col[str(cls)]] = probs[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out = out / np.maximum(row_sum, 1e-12)
    if not np.isfinite(out).all():
        raise RuntimeError(f"non-finite calibrated probabilities from {type(model).__name__}")
    return out


def top_predictions(probs: np.ndarray, class_order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = probs.argmax(axis=1)
    return class_order[idx].astype(str), probs[np.arange(len(idx)), idx]


def multiclass_brier(probs: np.ndarray, y_true: pd.Series, class_order: np.ndarray) -> float:
    class_to_col = {str(c): i for i, c in enumerate(class_order)}
    y = np.zeros_like(probs)
    for i, label in enumerate(y_true.astype(str)):
        j = class_to_col.get(str(label))
        if j is not None:
            y[i, j] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, model_name: str, pool: str) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(RELIABILITY_BINS[:-1], RELIABILITY_BINS[1:]):
        if hi == 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        rows.append(
            {
                "model": model_name,
                "pool": pool,
                "bin_low": float(lo),
                "bin_high": float(hi),
                "bin_mid": float((lo + hi) / 2),
                "n": n,
                "mean_confidence": float(conf[mask].mean()) if n else np.nan,
                "accuracy": float(correct[mask].mean()) if n else np.nan,
                "abs_gap": float(abs(conf[mask].mean() - correct[mask].mean())) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece_from_bins(bins: pd.DataFrame, n_total: int) -> float:
    usable = bins.dropna(subset=["abs_gap"]).copy()
    if usable.empty or n_total == 0:
        return float("nan")
    return float(((usable["n"] / n_total) * usable["abs_gap"]).sum())


def risk_coverage_rows(conf: np.ndarray, correct: np.ndarray, model_name: str, pool: str) -> pd.DataFrame:
    order = np.argsort(-conf)
    rows = []
    n = len(conf)
    for coverage in RISK_COVERAGES:
        k = max(1, int(np.ceil(n * coverage)))
        take = order[:k]
        rows.append(
            {
                "model": model_name,
                "pool": pool,
                "coverage": float(k / n),
                "n_covered": int(k),
                "accuracy_among_covered": float(correct[take].mean()),
                "error_among_covered": float(1.0 - correct[take].mean()),
                "min_confidence_covered": float(conf[take].min()),
                "mean_confidence_covered": float(conf[take].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_internal(models: Dict[str, object], x_test: np.ndarray, y_test: pd.Series, class_order: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_rows = []
    metric_rows = []
    bin_frames = []
    risk_frames = []
    for model_name, model in models.items():
        probs = aligned_predict_proba(model, x_test, class_order)
        pred, conf = top_predictions(probs, class_order)
        correct = pred == y_test.astype(str).to_numpy()
        pred_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "y_true": y_test.astype(str).to_numpy(),
                    "y_pred": pred,
                    "confidence": conf,
                    "correct": correct,
                }
            )
        )
        bins = reliability_bins(conf, correct.astype(float), model_name, "internal_test")
        bin_frames.append(bins)
        risk_frames.append(risk_coverage_rows(conf, correct.astype(float), model_name, "internal_test"))
        metric_rows.append(
            {
                "model": model_name,
                "pool": "internal_test",
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
                "mean_confidence": float(conf.mean()),
                "ece_top_label": ece_from_bins(bins, len(conf)),
                "multiclass_brier": multiclass_brier(probs, y_test, class_order),
                "temperature": float(getattr(model, "temperature", np.nan)),
                "n": int(len(conf)),
            }
        )
    return (
        pd.concat(pred_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.concat(bin_frames, ignore_index=True),
        pd.concat(risk_frames, ignore_index=True),
    )


def evaluate_external_pool(
    models: Dict[str, object],
    x: np.ndarray,
    details: pd.DataFrame,
    pool: str,
    class_order: np.ndarray,
    keep: set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frames = []
    summary_rows = []
    threshold_rows = []
    for model_name, model in models.items():
        probs = aligned_predict_proba(model, x, class_order)
        pred, conf = top_predictions(probs, class_order)
        out = details.copy()
        out.insert(0, "model", model_name)
        out.insert(1, "pool", pool)
        out["closed_set_disease_family"] = pred
        out["closed_set_calibrated_confidence"] = conf
        out["expected_disease_family_mapped"] = map_expected_family(out["expected_disease_family"], keep)
        out["closed_set_expected_match_raw"] = out["closed_set_disease_family"].astype(str) == out["expected_disease_family"].astype(str)
        out["closed_set_expected_match_train_vocab"] = out["closed_set_disease_family"].astype(str) == out["expected_disease_family_mapped"].astype(str)
        out["openworld_status"] = out["semantic_disease_semantic_status"].astype(str)
        out["openworld_stable"] = out["openworld_status"] == "stable"
        out["closed_set_over_openworld_mixed_or_unsupported"] = ~out["openworld_stable"]
        for threshold in CONFIDENCE_THRESHOLDS:
            out[f"closed_set_highconf_overcall_{int(threshold * 100):02d}"] = (
                out["closed_set_calibrated_confidence"] >= threshold
            ) & out["closed_set_over_openworld_mixed_or_unsupported"]
        pred_frames.append(out)
        summary_rows.append(
            {
                "model": model_name,
                "pool": pool,
                "n": int(len(out)),
                "mean_confidence": float(out["closed_set_calibrated_confidence"].mean()),
                "closed_set_expected_match_raw": float(out["closed_set_expected_match_raw"].mean()),
                "closed_set_expected_match_train_vocab": float(out["closed_set_expected_match_train_vocab"].mean()),
                "openworld_stable_rate": float(out["openworld_stable"].mean()),
                "openworld_mixed_or_unsupported_rate": float((~out["openworld_stable"]).mean()),
                "highconf_overcall_50": float(out["closed_set_highconf_overcall_50"].mean()),
                "highconf_overcall_70": float(out["closed_set_highconf_overcall_70"].mean()),
                "highconf_overcall_90": float(out["closed_set_highconf_overcall_90"].mean()),
            }
        )
        for threshold in CONFIDENCE_THRESHOLDS:
            covered = out["closed_set_calibrated_confidence"] >= threshold
            n_covered = int(covered.sum())
            if n_covered:
                threshold_rows.append(
                    {
                        "model": model_name,
                        "pool": pool,
                        "confidence_threshold": threshold,
                        "coverage": float(covered.mean()),
                        "n_covered": n_covered,
                        "overcall_among_covered_vs_openworld": float(
                            (covered & out["closed_set_over_openworld_mixed_or_unsupported"]).sum() / n_covered
                        ),
                        "expected_match_raw_among_covered": float(out.loc[covered, "closed_set_expected_match_raw"].mean()),
                        "expected_match_train_vocab_among_covered": float(out.loc[covered, "closed_set_expected_match_train_vocab"].mean()),
                    }
                )
            else:
                threshold_rows.append(
                    {
                        "model": model_name,
                        "pool": pool,
                        "confidence_threshold": threshold,
                        "coverage": 0.0,
                        "n_covered": 0,
                        "overcall_among_covered_vs_openworld": np.nan,
                        "expected_match_raw_among_covered": np.nan,
                        "expected_match_train_vocab_among_covered": np.nan,
                    }
                )
    return pd.concat(pred_frames, ignore_index=True), pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def svg_reliability(bins: pd.DataFrame, path: Path) -> None:
    width, height = 860, 410
    left, top, plot_w, plot_h = 72, 58, 610, 270
    colors = {
        "uncalibrated_sgd": "#1565C0",
        "temperature_scaled_sgd": "#7B1FA2",
        "sigmoid_calibrated": "#D9822B",
        "isotonic_calibrated": "#2E7D32",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Internal test top-label reliability after validation calibration</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#999" stroke-dasharray="5 4"/>',
    ]

    def xmap(v: float) -> float:
        return left + v * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - v * plot_h

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{xmap(tick):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.2f}</text>')
        parts.append(f'<text x="{left - 10}" y="{ymap(tick) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for model, sub in bins.dropna(subset=["mean_confidence", "accuracy"]).groupby("model"):
        sub = sub.sort_values("bin_mid")
        pts = [f"{xmap(float(r.mean_confidence)):.1f},{ymap(float(r.accuracy)):.1f}" for r in sub.itertuples(index=False)]
        if not pts:
            continue
        color = colors.get(model, "#555")
        parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for r in sub.itertuples(index=False):
            radius = max(3, min(9, np.sqrt(float(r.n)) / 2.2))
            parts.append(f'<circle cx="{xmap(float(r.mean_confidence)):.1f}" cy="{ymap(float(r.accuracy)):.1f}" r="{radius:.1f}" fill="{color}" opacity="0.82"/>')
    lx = width - 160
    for i, (model, color) in enumerate(colors.items()):
        y = 76 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(model)}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 22}" font-family="Arial" font-size="13" text-anchor="middle">mean confidence in bin</text>')
    parts.append('<text x="18" y="220" font-family="Arial" font-size="13" transform="rotate(-90 18 220)" text-anchor="middle">empirical accuracy</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_external_thresholds(thresholds: pd.DataFrame, path: Path) -> None:
    width, height = 980, 460
    left, top, plot_w, plot_h = 78, 62, 700, 292
    colors = {
        "uncalibrated_sgd": "#1565C0",
        "temperature_scaled_sgd": "#7B1FA2",
        "sigmoid_calibrated": "#D9822B",
        "isotonic_calibrated": "#2E7D32",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700">Calibrated confidence thresholding does not remove external closed-set overcall</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]

    def xmap(v: float) -> float:
        return left + v / 0.9 * plot_w

    def ymap(v: float) -> float:
        return top + plot_h - v * plot_h

    for tick in CONFIDENCE_THRESHOLDS:
        parts.append(f'<text x="{xmap(tick):.1f}" y="{top + plot_h + 22}" font-family="Arial" font-size="11" text-anchor="middle">{tick:.1f}</text>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{left - 12}" y="{ymap(tick) + 4:.1f}" font-family="Arial" font-size="11" text-anchor="end">{tick:.2f}</text>')
    for model, sub in thresholds.groupby("model"):
        sub = sub.groupby("confidence_threshold", as_index=False)["overcall_among_covered_vs_openworld"].mean().sort_values("confidence_threshold")
        pts = []
        for r in sub.itertuples(index=False):
            if pd.isna(r.overcall_among_covered_vs_openworld):
                continue
            pts.append(f"{xmap(float(r.confidence_threshold)):.1f},{ymap(float(r.overcall_among_covered_vs_openworld)):.1f}")
            parts.append(f'<circle cx="{xmap(float(r.confidence_threshold)):.1f}" cy="{ymap(float(r.overcall_among_covered_vs_openworld)):.1f}" r="4" fill="{colors.get(model, "#555")}"/>')
        if pts:
            parts.append(f'<path d="M {" L ".join(pts)}" fill="none" stroke="{colors.get(model, "#555")}" stroke-width="3"/>')
    lx = width - 176
    for i, (model, color) in enumerate(colors.items()):
        y = 76 + i * 24
        parts.append(f'<rect x="{lx}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{y}" font-family="Arial" font-size="12">{html.escape(model)}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 28}" font-family="Arial" font-size="13" text-anchor="middle">confidence threshold</text>')
    parts.append('<text x="18" y="230" font-family="Arial" font-size="13" transform="rotate(-90 18 230)" text-anchor="middle">overcall among covered samples</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary(internal_metrics: pd.DataFrame, external_summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    lines = [
        "# Calibrated predefined disease-label baseline",
        "",
        "## Purpose",
        "",
        "This analysis repeats the closed-set disease-family baseline with validation-set calibration. The classifier is trained on the frozen train split, calibrated on the frozen validation split with temperature scaling, sigmoid calibration, and isotonic calibration, then evaluated on the frozen internal test split and the locked External-180 / MultiSource-450 pools.",
        "",
        "## Internal test calibration",
        "",
    ]
    for row in internal_metrics.itertuples(index=False):
        temp_text = ""
        if hasattr(row, "temperature") and np.isfinite(float(row.temperature)):
            temp_text = f", temperature `{float(row.temperature):.3f}`"
        lines.append(
            f"- `{row.model}`: accuracy `{row.accuracy:.4f}`, macro-F1 `{row.macro_f1:.4f}`, "
            f"mean confidence `{row.mean_confidence:.4f}`, ECE `{row.ece_top_label:.4f}`, Brier `{row.multiclass_brier:.4f}`{temp_text}"
        )
    lines.extend(["", "## External overcall after calibration", ""])
    for row in external_summary.itertuples(index=False):
        lines.append(
            f"- `{row.model}` on `{row.pool}`: expected raw match `{row.closed_set_expected_match_raw:.4f}`, "
            f"mean confidence `{row.mean_confidence:.4f}`, open-world mixed/unsupported `{row.openworld_mixed_or_unsupported_rate:.4f}`, "
            f"high-confidence overcall >=0.70 `{row.highconf_overcall_70:.4f}`"
        )
    lines.extend(["", "## Threshold behavior", ""])
    for row in thresholds.loc[thresholds["confidence_threshold"].isin([0.5, 0.7, 0.9])].itertuples(index=False):
        cov = row.coverage
        over = row.overcall_among_covered_vs_openworld
        over_text = "NA" if pd.isna(over) else f"{over:.4f}"
        lines.append(
            f"- `{row.model}` `{row.pool}` threshold `{row.confidence_threshold:.1f}`: coverage `{cov:.4f}`, overcall among covered `{over_text}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Calibration makes the confidence values interpretable as model probabilities on the internal distribution, but it does not change the structural limitation of a predefined disease classifier: every external profile still receives a disease-family label. The downstream comparison tests whether calibrated high-confidence external calls remain concentrated in samples that the RNA-language portrait readout resolves as mixed or unsupported.",
            "",
            "## Output files",
            "",
            "- `t2b_internal_test_predictions.csv`",
            "- `t2b_internal_metrics.csv`",
            "- `t2b_internal_reliability_bins.csv`",
            "- `t2b_internal_risk_coverage.csv`",
            "- `t2b_external_predictions.csv`",
            "- `t2b_external_pool_summary.csv`",
            "- `t2b_external_thresholds.csv`",
            "- `t2b_reliability_internal.svg`",
            "- `t2b_external_threshold_overcall.svg`",
            "",
        ]
    )
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t2 = load_module(T2_SCRIPT, "t2b_helpers")
    train_module = t2.load_module(CODE_DIR / "train_rna_language_alignment.py", "t2b_train_module")
    taxonomy = t2.load_module(CODE_DIR / "family_taxonomy.py", "t2b_family_taxonomy")
    ckpt = t2.torch_load(t2.CHECKPOINT)

    x_train, train_meta = t2.prepare_training_matrix(train_module, ckpt, split="train")
    x_val, val_meta = t2.prepare_training_matrix(train_module, ckpt, split="val")
    x_test, test_meta = t2.prepare_training_matrix(train_module, ckpt, split="test")
    y_train, keep = make_labels_with_train_vocab(train_meta, taxonomy)
    y_val, _ = make_labels_with_train_vocab(val_meta, taxonomy, keep)
    y_test, _ = make_labels_with_train_vocab(test_meta, taxonomy, keep)

    base = train_base_classifier(x_train, y_train)
    class_order = np.asarray(base.classes_).astype(str)
    temperature = fit_temperature_scaled(base, x_val, y_val, class_order)
    sigmoid = calibrate_prefit(base, x_val, y_val, "sigmoid")
    isotonic = calibrate_prefit(base, x_val, y_val, "isotonic")
    models = {
        "uncalibrated_sgd": base,
        "temperature_scaled_sgd": temperature,
        "sigmoid_calibrated": sigmoid,
        "isotonic_calibrated": isotonic,
    }

    internal_pred, internal_metrics, reliability, risk = evaluate_internal(models, x_test, y_test, class_order)
    internal_pred.to_csv(OUTDIR / "t2b_internal_test_predictions.csv", index=False)
    internal_metrics.to_csv(OUTDIR / "t2b_internal_metrics.csv", index=False)
    reliability.to_csv(OUTDIR / "t2b_internal_reliability_bins.csv", index=False)
    risk.to_csv(OUTDIR / "t2b_internal_risk_coverage.csv", index=False)

    ext_preds = []
    ext_summaries = []
    ext_thresholds = []
    for pool, paths in t2.POOLS.items():
        details = pd.read_csv(paths["details"])
        x_ext, ext_details = t2.prepare_external_matrix(details, paths["inputs"], ckpt)
        pred, summary, thresholds = evaluate_external_pool(models, x_ext, ext_details, pool, class_order, keep)
        ext_preds.append(pred)
        ext_summaries.append(summary)
        ext_thresholds.append(thresholds)
    external_pred = pd.concat(ext_preds, ignore_index=True)
    external_summary = pd.concat(ext_summaries, ignore_index=True)
    external_thresholds = pd.concat(ext_thresholds, ignore_index=True)
    external_pred.to_csv(OUTDIR / "t2b_external_predictions.csv", index=False)
    external_summary.to_csv(OUTDIR / "t2b_external_pool_summary.csv", index=False)
    external_thresholds.to_csv(OUTDIR / "t2b_external_thresholds.csv", index=False)

    svg_reliability(reliability, OUTDIR / "t2b_reliability_internal.svg")
    svg_external_thresholds(external_thresholds, OUTDIR / "t2b_external_threshold_overcall.svg")
    write_summary(internal_metrics, external_summary, external_thresholds)


if __name__ == "__main__":
    main()
