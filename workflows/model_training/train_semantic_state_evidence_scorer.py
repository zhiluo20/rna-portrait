#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from repro_paths import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DETAILS = OUTPUT_ROOT / "multisource_450_trimmed_benchmark" / "details.csv"
DEFAULT_OUTDIR = OUTPUT_ROOT / "semantic_state_evidence_scorer_max"
EVIDENCE_KINDS = [
    "semantic_majority",
    "anchor_context",
    "neighbor_context",
    "age_driver_overlap",
    "tension",
    "anchor_override",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return ap.parse_args()


def _row_to_features(row: pd.Series) -> dict[str, object]:
    evidence = json.loads(row.get("semantic_state_evidence_json", "[]") or "[]")
    feat: dict[str, object] = {
        "state_family": str(row.get("semantic_state_family", "unknown")),
        "state_subprofile": str(row.get("semantic_state_subprofile", "unknown")),
        "route": str(row.get("semantic_route", "unknown")),
        "mode": str(row.get("semantic_explanation_mode", "unknown")),
        "anchor_adjusted": int(row.get("semantic_anchor_adjusted", 0) or 0),
    }
    for kind in EVIDENCE_KINDS:
        item = next((x for x in evidence if str(x.get("kind")) == kind), {})
        feat[f"strength_{kind}"] = float(item.get("strength", 0.0) or 0.0)
    return feat


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.details)
    if df.empty:
        raise SystemExit("empty details.csv")

    x = [_row_to_features(row) for _, row in df.iterrows()]
    y = df["semantic_state_top_evidence_kind"].astype(str)
    groups = df["project"].astype(str) if "project" in df.columns else None

    pipe = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    min_class = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = cross_val_score(pipe, x, y, cv=cv, scoring="accuracy")
    group_acc = None
    group_splits = None
    if groups is not None and groups.nunique() >= 3:
        group_splits = max(2, min(5, int(groups.nunique())))
        group_cv = GroupKFold(n_splits=group_splits)
        group_acc = cross_val_score(pipe, x, y, cv=group_cv, groups=groups, scoring="accuracy")
    pipe.fit(x, y)
    pred = pipe.predict(x)
    with open(outdir / "model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    summary = {
        "n_samples": int(len(df)),
        "label_counts": y.value_counts().to_dict(),
        "cv_splits": int(n_splits),
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "group_cv_splits": int(group_splits) if group_splits is not None else None,
        "group_cv_accuracy_mean": float(group_acc.mean()) if group_acc is not None else None,
        "group_cv_accuracy_std": float(group_acc.std()) if group_acc is not None else None,
        "train_accuracy": float(accuracy_score(y, pred)),
        "model_path": "model.pkl",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (outdir / "summary.md").write_text(
        "\n".join(
            [
                "# Semantic State Evidence Scorer",
                "",
                f"- n_samples: `{summary['n_samples']}`",
                f"- label_counts: `{summary['label_counts']}`",
                f"- cv_splits: `{summary['cv_splits']}`",
                f"- cv_accuracy_mean: `{summary['cv_accuracy_mean']:.4f}`",
                f"- cv_accuracy_std: `{summary['cv_accuracy_std']:.4f}`",
                f"- group_cv_splits: `{summary['group_cv_splits']}`",
                f"- group_cv_accuracy_mean: `{summary['group_cv_accuracy_mean']:.4f}`" if summary["group_cv_accuracy_mean"] is not None else "- group_cv_accuracy_mean: `null`",
                f"- group_cv_accuracy_std: `{summary['group_cv_accuracy_std']:.4f}`" if summary["group_cv_accuracy_std"] is not None else "- group_cv_accuracy_std: `null`",
                f"- train_accuracy: `{summary['train_accuracy']:.4f}`",
            ]
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
