from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from runtime_paths import SUPP_DIR


RESULTS = SUPP_DIR
OUT = SUPP_DIR / "T13_portrait_composition_disease_information"
RANDOM_STATE = 20260610


STATE_ORDER = [
    "stable_consensus",
    "hematologic_override",
    "epithelial_override",
    "clean_anchor_override",
    "generic_context_override",
    "unsupported_semantics",
    "family_conflict",
    "other",
]

STATE_LABELS = {
    "stable_consensus": "single clear signal",
    "hematologic_override": "blood/immune",
    "epithelial_override": "epithelial-like context",
    "clean_anchor_override": "cleaner anchor context",
    "generic_context_override": "broad context",
    "unsupported_semantics": "weak evidence",
    "family_conflict": "conflict",
    "other": "other",
}

STATE_COLORS = {
    "stable_consensus": "#303030",
    "hematologic_override": "#1F5A9D",
    "epithelial_override": "#B64A4A",
    "clean_anchor_override": "#5B8C5A",
    "generic_context_override": "#7E6AAE",
    "unsupported_semantics": "#A9A9A9",
    "family_conflict": "#C9822B",
    "other": "#6B6B6B",
}

PALETTE = {
    "blue": "#1F5A9D",
    "blue_light": "#9DB9D8",
    "green": "#5B8C5A",
    "green_light": "#B8D3B4",
    "red": "#B64A4A",
    "orange": "#C9822B",
    "orange_light": "#E1B66D",
    "violet": "#7E6AAE",
    "grey": "#8F8F8F",
    "grey_light": "#D9D9D9",
    "black": "#222222",
}


MARKER_FEATURES = [
    "immune_core_z",
    "t_cell_nk_z",
    "myeloid_inflammation_z",
    "hematologic_lineage_z",
    "epithelial_z",
    "stromal_ecm_z",
    "proliferation_z",
    "interferon_z",
]

PATHWAY_FEATURES = [
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
    "oxidative_phosphorylation_score",
]

CONTROL_FEATURES = [
    "expected_site_family",
    "source_batch_proxy",
    "project_prefix",
    "pool",
]

PORTRAIT_FEATURES = [
    "semantic_state_family",
]

PORTRAIT_CONTEXT_FEATURES = [
    "semantic_state_family",
    "anchor_context_state",
    "semantic_state_top_evidence_kind",
]


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> dict[str, str]:
    paths = {}
    for ext in ["svg", "pdf", "png"]:
        path = OUT / f"{name}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if ext == "png":
            kwargs["dpi"] = 360
        fig.savefig(path, **kwargs)
        paths[ext] = str(path)
    plt.close(fig)
    return paths


def wrap_label(text: str, width: int = 18) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(str(text).replace("_", " "), width=width, break_long_words=False))


def load_data() -> pd.DataFrame:
    path = RESULTS / "T8_shortcut_exclusion_controls/t8_merged_shortcut_input.csv"
    df = pd.read_csv(path)
    df = df.copy()
    df["disease_label"] = df["expected_disease_family"].astype(str)
    df["portrait_family"] = df["semantic_state_family"].astype(str)
    df["portrait_label"] = df["portrait_family"].map(STATE_LABELS).fillna(df["portrait_family"])
    df["project_prefix"] = df["project_prefix"].fillna(df["project"].astype(str).str.split("_").str[0])
    for col in CONTROL_FEATURES + PORTRAIT_CONTEXT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def filter_diseases(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    counts = df["disease_label"].value_counts()
    keep = counts[counts >= min_n].index
    return df.loc[df["disease_label"].isin(keep)].copy().reset_index(drop=True)


def portrait_composition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.groupby(["disease_label", "portrait_family"])
        .size()
        .reset_index(name="n")
    )
    count_matrix = (
        counts.pivot(index="disease_label", columns="portrait_family", values="n")
        .reindex(columns=STATE_ORDER, fill_value=0)
        .fillna(0)
        .astype(int)
    )
    frac = count_matrix.div(count_matrix.sum(axis=1), axis=0).fillna(0)
    long = (
        count_matrix.stack()
        .rename("n")
        .reset_index()
    )
    long["fraction"] = long.apply(
        lambda r: frac.loc[r["disease_label"], r["portrait_family"]],
        axis=1,
    )
    long["portrait_label"] = long["portrait_family"].map(STATE_LABELS).fillna(long["portrait_family"])
    long["disease_n"] = long["disease_label"].map(count_matrix.sum(axis=1).to_dict())
    return long, frac


def composition_summary(frac: pd.DataFrame, counts_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    totals = counts_long.groupby("disease_label")["disease_n"].first()
    for disease, row in frac.iterrows():
        p = row.to_numpy(dtype=float)
        p = p / p.sum() if p.sum() else p
        top_idx = int(np.argmax(p))
        effective = float(np.exp(entropy(p + 1e-12)))
        rows.append(
            {
                "disease_label": disease,
                "n": int(totals.loc[disease]),
                "top_portrait_family": row.index[top_idx],
                "top_portrait_label": STATE_LABELS.get(row.index[top_idx], row.index[top_idx]),
                "top_portrait_fraction": float(p[top_idx]),
                "effective_portrait_count": effective,
                "entropy": float(entropy(p + 1e-12)),
            }
        )
    return pd.DataFrame(rows).sort_values(["n", "disease_label"], ascending=[False, True])


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a + 1e-9, b + 1e-9, base=2.0))


def repeated_group_composition_identification(df: pd.DataFrame, n_repeats: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    diseases = sorted(df["disease_label"].unique())
    rows = []
    for rep in range(n_repeats):
        train_parts = []
        test_parts = []
        for disease in diseases:
            sub = df.loc[df["disease_label"].eq(disease)]
            idx = sub.index.to_numpy()
            rng.shuffle(idx)
            n_train = max(5, int(round(len(idx) * 0.6)))
            n_train = min(n_train, len(idx) - 3)
            train_parts.append(df.loc[idx[:n_train]])
            test_parts.append(df.loc[idx[n_train:]])
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        _, train_frac = portrait_composition(train)
        _, test_frac = portrait_composition(test)
        for disease in diseases:
            observed = test_frac.loc[disease].to_numpy(dtype=float)
            distances = {
                candidate: js_distance(observed, train_frac.loc[candidate].to_numpy(dtype=float))
                for candidate in diseases
            }
            pred = min(distances, key=distances.get)
            sorted_dist = sorted(distances.items(), key=lambda kv: kv[1])
            rows.append(
                {
                    "repeat": rep,
                    "disease_label": disease,
                    "predicted_disease": pred,
                    "correct": int(pred == disease),
                    "own_js_distance": distances[disease],
                    "nearest_other_disease": next(k for k, _ in sorted_dist if k != disease),
                    "nearest_other_js_distance": next(v for k, v in sorted_dist if k != disease),
                }
            )
    return pd.DataFrame(rows)


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=None), categorical),
            ("num", numeric_pipeline, numeric),
        ],
        remainder="drop",
    )


def make_model(categorical: list[str], numeric: list[str]) -> Pipeline:
    pre = make_preprocessor(categorical, numeric)
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", pre), ("classifier", clf)])


@dataclass
class ModelSpec:
    name: str
    label: str
    categorical: list[str]
    numeric: list[str]


def evaluate_models(df: pd.DataFrame, model_specs: list[ModelSpec], n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.copy()
    y = X["disease_label"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    pred_rows = []
    labels = sorted(y.unique())

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE + fold)
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        dummy_pred = dummy.predict(np.zeros((len(test_idx), 1)))
        fold_rows.append(
            {
                "model": "stratified_dummy",
                "model_label": "stratified dummy",
                "fold": fold,
                "accuracy": accuracy_score(y_test, dummy_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, dummy_pred),
                "macro_f1": f1_score(y_test, dummy_pred, average="macro"),
                "log_loss": np.nan,
            }
        )

        comp_pred = composition_centroid_predict(X_train, X_test)
        fold_rows.append(
            {
                "model": "portrait_composition_centroid",
                "model_label": "portrait composition centroid",
                "fold": fold,
                "accuracy": accuracy_score(y_test, comp_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, comp_pred),
                "macro_f1": f1_score(y_test, comp_pred, average="macro"),
                "log_loss": np.nan,
            }
        )

        for spec in model_specs:
            model = make_model(spec.categorical, spec.numeric)
            model.fit(X_train[spec.categorical + spec.numeric], y_train)
            pred = model.predict(X_test[spec.categorical + spec.numeric])
            try:
                proba = model.predict_proba(X_test[spec.categorical + spec.numeric])
                model_labels = list(model.classes_)
                ll = log_loss(y_test, proba, labels=model_labels)
            except Exception:
                ll = np.nan
            fold_rows.append(
                {
                    "model": spec.name,
                    "model_label": spec.label,
                    "fold": fold,
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro"),
                    "log_loss": ll,
                }
            )
            pred_rows.extend(
                {
                    "model": spec.name,
                    "fold": fold,
                    "sample_id": X_test.iloc[i].get("sample_id", X_test.iloc[i].get("file", "")),
                    "true_disease": y_test.iloc[i],
                    "predicted_disease": pred[i],
                }
                for i in range(len(X_test))
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def composition_centroid_predict(train: pd.DataFrame, test: pd.DataFrame, alpha: float = 1.0) -> list[str]:
    diseases = sorted(train["disease_label"].unique())
    count = (
        train.groupby(["disease_label", "portrait_family"]).size().unstack(fill_value=0)
        .reindex(index=diseases, columns=STATE_ORDER, fill_value=0)
        .astype(float)
    )
    probs = (count + alpha).div((count + alpha).sum(axis=1), axis=0)
    pred = []
    for _, row in test.iterrows():
        state = row["portrait_family"]
        if state not in probs.columns:
            pred.append(diseases[0])
            continue
        # Uniform disease prior: this asks whether the portrait composition itself distinguishes diseases.
        pred.append(str(probs[state].idxmax()))
    return pred


def permuted_portrait_control_test(df: pd.DataFrame, n_permutations: int = 80) -> pd.DataFrame:
    real_specs = [
        ModelSpec("controls", "metadata controls", CONTROL_FEATURES, []),
        ModelSpec("controls_plus_portrait", "controls + portrait family", CONTROL_FEATURES + PORTRAIT_FEATURES, []),
    ]
    real_fold, _ = evaluate_models(df, real_specs)
    real_delta = (
        real_fold.groupby("model")["balanced_accuracy"].mean().loc["controls_plus_portrait"]
        - real_fold.groupby("model")["balanced_accuracy"].mean().loc["controls"]
    )

    rng = np.random.default_rng(RANDOM_STATE + 100)
    rows = [{"test": "real_increment_over_controls", "iteration": -1, "delta_balanced_accuracy": real_delta}]
    strata = df[["expected_site_family", "source_batch_proxy"]].astype(str).agg("||".join, axis=1)
    for i in range(n_permutations):
        perm = df.copy()
        shuffled = perm["portrait_family"].copy()
        for _, idx in strata.groupby(strata).groups.items():
            idx = list(idx)
            vals = shuffled.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuffled.loc[idx] = vals
        perm["portrait_family"] = shuffled
        perm["semantic_state_family"] = shuffled
        fold, _ = evaluate_models(perm, real_specs)
        delta = (
            fold.groupby("model")["balanced_accuracy"].mean().loc["controls_plus_portrait"]
            - fold.groupby("model")["balanced_accuracy"].mean().loc["controls"]
        )
        rows.append({"test": "within_tissue_source_permutation", "iteration": i, "delta_balanced_accuracy": delta})
    out = pd.DataFrame(rows)
    null = out.loc[out["test"].eq("within_tissue_source_permutation"), "delta_balanced_accuracy"]
    p = (1 + (null >= real_delta).sum()) / (len(null) + 1)
    out["one_sided_p_value_for_real_delta"] = p
    return out


def source_tissue_stability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    _, overall = portrait_composition(df)
    for disease, sub in df.groupby("disease_label"):
        disease_n = len(sub)
        for factor in ["source_batch_proxy", "project_prefix", "expected_site_family"]:
            for level, g in sub.groupby(factor):
                if len(g) < 5:
                    continue
                _, frac = portrait_composition(g)
                rows.append(
                    {
                        "disease_label": disease,
                        "disease_n": disease_n,
                        "control_factor": factor,
                        "control_level": level,
                        "level_n": len(g),
                        "js_distance_to_disease_overall": js_distance(
                            frac.loc[disease].to_numpy(dtype=float),
                            overall.loc[disease].to_numpy(dtype=float),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def draw_composition_heatmap(frac: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    order = summary.sort_values("n", ascending=False)["disease_label"].tolist()
    plot = frac.loc[order, STATE_ORDER]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#DCE8F3", PALETTE["blue"]])
    im = ax.imshow(plot.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=max(0.6, float(plot.max().max())))
    ax.set_yticks(np.arange(len(plot)), [wrap_label(f"{d} (n={int(summary.set_index('disease_label').loc[d, 'n'])})", 23) for d in plot.index])
    ax.set_xticks(np.arange(len(STATE_ORDER)), [wrap_label(STATE_LABELS[s], 13) for s in STATE_ORDER], rotation=35, ha="right")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("fraction within disease label")
    ax.set_title("Disease labels have distinct portrait compositions")
    return save_figure(fig, "t13_disease_portrait_composition_heatmap")


def draw_model_comparison(perf_summary: pd.DataFrame) -> dict[str, str]:
    order = [
        "stratified_dummy",
        "portrait_composition_centroid",
        "portrait_family",
        "metadata_controls",
        "controls_plus_portrait",
        "marker_pathway",
        "controls_plus_marker_pathway",
        "controls_plus_portrait_marker_pathway",
    ]
    label_map = perf_summary.set_index("model")["model_label"].to_dict()
    plot = perf_summary.set_index("model").loc[[m for m in order if m in perf_summary["model"].values]].reset_index()
    y = np.arange(len(plot))[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    colors = [PALETTE["grey"] if "dummy" in m else PALETTE["blue"] if "portrait" in m else PALETTE["green"] if "marker" in m else PALETTE["orange"] for m in plot["model"]]
    ax.barh(y, plot["balanced_accuracy_mean"], color=colors, xerr=plot["balanced_accuracy_sem"], zorder=3)
    ax.set_yticks(y, [wrap_label(label_map.get(m, m), 26) for m in plot["model"]])
    ax.set_xlabel("cross-validated balanced accuracy")
    ax.set_xlim(0, max(0.35, float((plot["balanced_accuracy_mean"] + plot["balanced_accuracy_sem"]).max()) * 1.22))
    ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_title("Disease prediction from portrait and molecular features")
    return save_figure(fig, "t13_disease_prediction_model_comparison")


def draw_control_permutation(permutation: pd.DataFrame) -> dict[str, str]:
    real = permutation.loc[permutation["test"].eq("real_increment_over_controls"), "delta_balanced_accuracy"].iloc[0]
    null = permutation.loc[permutation["test"].eq("within_tissue_source_permutation"), "delta_balanced_accuracy"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(null, bins=18, color=PALETTE["grey_light"], edgecolor="#777777", linewidth=0.5)
    ax.axvline(real, color=PALETTE["red"], lw=1.8, label="real portrait increment")
    ax.set_xlabel("balanced-accuracy gain over metadata controls")
    ax.set_ylabel("permutations")
    ax.set_title("Portrait signal after tissue/source controls")
    ax.legend(loc="upper right")
    return save_figure(fig, "t13_portrait_increment_after_controls_permutation")


def draw_group_identification(group_id: pd.DataFrame) -> dict[str, str]:
    summary = (
        group_id.groupby("disease_label")
        .agg(
            composition_id_accuracy=("correct", "mean"),
            own_js_distance=("own_js_distance", "mean"),
            nearest_other_js_distance=("nearest_other_js_distance", "mean"),
        )
        .reset_index()
        .sort_values("composition_id_accuracy")
    )
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh(y, summary["composition_id_accuracy"], color=PALETTE["violet"], zorder=3)
    ax.set_yticks(y, [wrap_label(x, 24) for x in summary["disease_label"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("nearest-composition disease identification accuracy")
    ax.grid(axis="x", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_title("Disease-level portrait compositions are partly identifiable")
    save_figure(fig, "t13_group_level_composition_identification")
    return {"summary_table": str(OUT / "t13_group_level_composition_identification_by_disease.csv")}


def write_summary(metrics: dict[str, object]) -> None:
    lines = [
        "# Portrait composition and disease-label information",
        "",
        "## Question",
        "",
        "Can disease labels be represented by characteristic mixtures of molecular portrait families, and does that mixture carry disease-discriminative information after source, tissue and platform controls?",
        "",
        "## Inputs",
        "",
        "- sample-level portrait calls, metadata controls and marker/pathway scores from the source/metadata-control analysis.",
        "- Disease label: `expected_disease_family`.",
        "- Portrait family: `semantic_state_family`, displayed with reader-facing labels.",
        "- Controls: `expected_site_family`, `source_batch_proxy`, `project_prefix` and `pool`.",
        "",
        "## Main results",
        "",
        f"- Analysed samples: {metrics['n_samples']} across {metrics['n_diseases']} disease labels with at least {metrics['min_n']} samples.",
        f"- Disease-level composition identification accuracy: {metrics['group_identification_accuracy']:.3f}.",
        f"- Portrait-composition centroid balanced accuracy: {metrics['composition_centroid_balanced_accuracy']:.3f}.",
        f"- Portrait-family logistic balanced accuracy: {metrics['portrait_balanced_accuracy']:.3f}.",
        f"- Metadata-control balanced accuracy: {metrics['controls_balanced_accuracy']:.3f}.",
        f"- Metadata controls plus portrait-family balanced accuracy: {metrics['controls_plus_portrait_balanced_accuracy']:.3f}.",
        f"- Increment over controls: {metrics['portrait_increment_over_controls']:.3f}; within-tissue/source permutation one-sided P = {metrics['portrait_increment_permutation_p']:.4f}.",
        f"- Marker/pathway balanced accuracy: {metrics['marker_pathway_balanced_accuracy']:.3f}.",
        f"- Metadata controls plus marker/pathway balanced accuracy: {metrics['controls_plus_marker_pathway_balanced_accuracy']:.3f}.",
        f"- Metadata controls plus portrait plus marker/pathway balanced accuracy: {metrics['full_model_balanced_accuracy']:.3f}.",
        "",
        "## Interpretation",
        "",
        metrics["interpretation"],
        "",
        "## Output files",
        "",
        "- `t13_disease_portrait_composition_long.csv`",
        "- `t13_disease_portrait_composition_matrix.csv`",
        "- `t13_disease_portrait_composition_summary.csv`",
        "- `t13_model_fold_metrics.csv`",
        "- `t13_model_performance_summary.csv`",
        "- `t13_source_tissue_stability.csv`",
        "- `t13_within_tissue_source_permutation.csv`",
        "- `t13_group_level_composition_identification.csv`",
        "- `t13_disease_portrait_composition_heatmap.svg/pdf/png`",
        "- `t13_disease_prediction_model_comparison.svg/pdf/png`",
        "- `t13_portrait_increment_after_controls_permutation.svg/pdf/png`",
        "- `t13_group_level_composition_identification.svg/pdf/png`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_style()
    df_all = load_data()
    min_n = 10
    df = filter_diseases(df_all, min_n=min_n)
    df.to_csv(OUT / "t13_analysis_sample_table.csv", index=False)
    marker_features = [c for c in MARKER_FEATURES if c in df.columns]
    pathway_features = [c for c in PATHWAY_FEATURES if c in df.columns]
    missing_features = sorted(set(MARKER_FEATURES + PATHWAY_FEATURES) - set(marker_features + pathway_features))
    (OUT / "t13_feature_audit.json").write_text(
        json.dumps(
            {
                "marker_features_used": marker_features,
                "pathway_features_used": pathway_features,
                "missing_requested_features": missing_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    comp_long, comp_frac = portrait_composition(df)
    comp_summary = composition_summary(comp_frac, comp_long)
    comp_long.to_csv(OUT / "t13_disease_portrait_composition_long.csv", index=False)
    comp_frac.rename(columns=STATE_LABELS).to_csv(OUT / "t13_disease_portrait_composition_matrix.csv")
    comp_summary.to_csv(OUT / "t13_disease_portrait_composition_summary.csv", index=False)

    group_id = repeated_group_composition_identification(df)
    group_id.to_csv(OUT / "t13_group_level_composition_identification.csv", index=False)
    group_by_disease = (
        group_id.groupby("disease_label")
        .agg(
            composition_id_accuracy=("correct", "mean"),
            own_js_distance=("own_js_distance", "mean"),
            nearest_other_js_distance=("nearest_other_js_distance", "mean"),
        )
        .reset_index()
    )
    group_by_disease.to_csv(OUT / "t13_group_level_composition_identification_by_disease.csv", index=False)

    model_specs = [
        ModelSpec("portrait_family", "portrait family only", PORTRAIT_FEATURES, []),
        ModelSpec("portrait_context", "portrait + context", PORTRAIT_CONTEXT_FEATURES, ["semantic_state_top_evidence_score", "semantic_state_second_evidence_score"]),
        ModelSpec("metadata_controls", "metadata controls", CONTROL_FEATURES, []),
        ModelSpec("controls_plus_portrait", "controls + portrait family", CONTROL_FEATURES + PORTRAIT_FEATURES, []),
        ModelSpec("marker_pathway", "marker + pathway signatures", [], marker_features + pathway_features),
        ModelSpec("controls_plus_marker_pathway", "controls + marker/pathway", CONTROL_FEATURES, marker_features + pathway_features),
        ModelSpec("controls_plus_portrait_marker_pathway", "controls + portrait + marker/pathway", CONTROL_FEATURES + PORTRAIT_FEATURES, marker_features + pathway_features),
    ]
    fold_metrics, predictions = evaluate_models(df, model_specs)
    fold_metrics.to_csv(OUT / "t13_model_fold_metrics.csv", index=False)
    predictions.to_csv(OUT / "t13_model_predictions.csv", index=False)
    perf = (
        fold_metrics.groupby(["model", "model_label"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sem=("accuracy", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_sem=("balanced_accuracy", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sem=("macro_f1", lambda x: float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0),
            log_loss_mean=("log_loss", "mean"),
        )
        .reset_index()
        .sort_values("balanced_accuracy_mean", ascending=False)
    )
    perf.to_csv(OUT / "t13_model_performance_summary.csv", index=False)

    stability = source_tissue_stability(df)
    stability.to_csv(OUT / "t13_source_tissue_stability.csv", index=False)

    permutation = permuted_portrait_control_test(df)
    permutation.to_csv(OUT / "t13_within_tissue_source_permutation.csv", index=False)

    draw_composition_heatmap(comp_frac, comp_summary)
    draw_model_comparison(perf)
    draw_control_permutation(permutation)
    draw_group_identification(group_id)

    perf_idx = perf.set_index("model")
    group_acc = float(group_id["correct"].mean())
    real_delta = float(permutation.loc[permutation["test"].eq("real_increment_over_controls"), "delta_balanced_accuracy"].iloc[0])
    p_perm = float(permutation["one_sided_p_value_for_real_delta"].iloc[0])
    if real_delta > 0.03 and p_perm < 0.05:
        interpretation = (
            "Disease labels are associated with characteristic molecular-portrait compositions, and portrait family adds "
            "disease-discriminative information beyond broad tissue/source/platform controls in this analysis."
        )
    elif real_delta > 0:
        interpretation = (
            "Portrait composition contains disease-related signal, but the incremental gain beyond tissue/source/platform controls "
            "is modest or not clearly stronger than the within-stratum permutation baseline."
        )
    else:
        interpretation = (
            "This analysis does not support the stronger claim that portrait composition adds disease-discriminative information "
            "beyond tissue/source/platform controls."
        )

    metrics = {
        "n_samples": int(len(df)),
        "n_diseases": int(df["disease_label"].nunique()),
        "min_n": min_n,
        "group_identification_accuracy": group_acc,
        "composition_centroid_balanced_accuracy": float(perf_idx.loc["portrait_composition_centroid", "balanced_accuracy_mean"]),
        "portrait_balanced_accuracy": float(perf_idx.loc["portrait_family", "balanced_accuracy_mean"]),
        "controls_balanced_accuracy": float(perf_idx.loc["metadata_controls", "balanced_accuracy_mean"]),
        "controls_plus_portrait_balanced_accuracy": float(perf_idx.loc["controls_plus_portrait", "balanced_accuracy_mean"]),
        "portrait_increment_over_controls": real_delta,
        "portrait_increment_permutation_p": p_perm,
        "marker_pathway_balanced_accuracy": float(perf_idx.loc["marker_pathway", "balanced_accuracy_mean"]),
        "controls_plus_marker_pathway_balanced_accuracy": float(perf_idx.loc["controls_plus_marker_pathway", "balanced_accuracy_mean"]),
        "full_model_balanced_accuracy": float(perf_idx.loc["controls_plus_portrait_marker_pathway", "balanced_accuracy_mean"]),
        "interpretation": interpretation,
    }
    (OUT / "t13_key_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
