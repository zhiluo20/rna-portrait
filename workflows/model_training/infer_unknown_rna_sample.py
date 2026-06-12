#!/usr/bin/env python3
"""Infer phenotype-like outputs for an unknown bulk RNA count profile.

This is intentionally an inference tool, not a clinical risk model.
It maps a pasted `gene,count` profile into the existing multimodal embedding,
then reports age/sex/tumor/site/disease-family style outputs plus evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
try:
    import joblib
except Exception:  # pragma: no cover - optional runtime dependency
    joblib = None

from family_taxonomy import disease_family, organ_family
from disease_label_normalization import normalize_disease_label
from repro_paths import BME_CODE_DIR as MODULE_DIR
from repro_paths import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[2]
EXPLAINER_ROUTER_PATH = OUTPUT_ROOT / "train_explainer_support_router_20260417" / "router.joblib"

DEFAULT_AGE_ADAPTER_RUN = "train_age_adapter_curated_ageq_batch18_sbert_allminilm_v2"
DEFAULT_TOP_K = 10
MIN_SELECTED_GENES_MATCHED = 32
MIN_SELECTED_GENE_COVERAGE = 0.01

ARCHETYPE_MARKERS = {
    "immune_dominant": ["PTPRC", "CD3D", "CD3E", "NKG7", "LST1", "TYROBP", "HLA-DRA", "CD74"],
    "epithelial_dominant": ["EPCAM", "KRT8", "KRT18", "KRT19", "MSLN", "KRT17"],
    "squamous_like": ["KRT5", "KRT14", "TP63", "SOX2", "DSG3"],
    "thoracic_like": ["NKX2-1", "SFTPA1", "SFTPA2", "SFTPB", "SFTPC"],
    "breast_like": ["ESR1", "PGR", "ERBB2", "GATA3", "FOXA1"],
    "hematologic_like": ["PTPRC", "MS4A1", "CD79A", "CD74", "HLA-DRA", "NKG7"],
}

ARCHETYPE_CONTEXT_LABELS = {
    "immune_dominant": "immune-dominant expression context",
    "hematologic_like": "hematologic-like expression context",
    "epithelial_dominant": "epithelial-like expression context",
    "squamous_like": "squamous-like expression context",
    "thoracic_like": "thoracic-like expression context",
    "breast_like": "breast-like expression context",
}

RELATIVE_SIGNATURE_MARKERS = {
    "breast_like": ["GATA3", "FOXA1", "ESR1", "PGR", "ERBB2", "KRT19"],
    "hematologic_like": ["PTPRC", "MS4A1", "CD79A", "CD74", "HLA-DRA", "NKG7"],
    "liver_like": ["ALB", "APOA1", "TTR", "APOC3", "FGB", "CYP3A4"],
    "brain_like": ["SNAP25", "RBFOX3", "SLC17A7", "GAD1", "MAP2", "SYT1"],
}


def _input_quality_tier(matched_genes: int, coverage: float) -> str:
    if matched_genes < MIN_SELECTED_GENES_MATCHED or coverage < MIN_SELECTED_GENE_COVERAGE:
        return "insufficient"
    if matched_genes < 128 or coverage < 0.03:
        return "sparse"
    if matched_genes < 512 or coverage < 0.10:
        return "adequate"
    return "strong"


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_explainer_router():
    if joblib is None or not EXPLAINER_ROUTER_PATH.exists():
        return None
    try:
        return joblib.load(EXPLAINER_ROUTER_PATH)
    except Exception:
        return None


def _predict_explainer_router_score(
    *,
    support_level: str,
    quality_tier: str,
    context_state: str,
    tissue_context: str,
    origin_hypothesis: str,
    support_confidence: float,
    thoracic_support: float,
) -> float | None:
    model = load_explainer_router()
    if model is None:
        return None
    frame = pd.DataFrame(
        [
            {
                "support_level": str(support_level or "unknown"),
                "explainer_quality_tier": str(quality_tier or "unknown"),
                "context_state": str(context_state or "unknown"),
                "tissue_context": str(tissue_context or "unknown"),
                "origin_hypothesis_present": "none" if str(origin_hypothesis or "none") in {"none", "unknown", ""} else "present",
                "support_confidence": float(support_confidence or 0.0),
                "thoracic_support": float(thoracic_support or 0.0),
            }
        ]
    )
    try:
        return float(model.predict_proba(frame)[0, 1])
    except Exception:
        return None


@dataclass
class UnknownSampleArtifacts:
    model: torch.nn.Module
    age_adapter: torch.nn.Module
    train_module: object
    checkpoint: dict
    selected_genes: List[str]
    expr_mean: np.ndarray
    expr_std: np.ndarray
    train_emb: np.ndarray
    train_meta: pd.DataFrame
    disease_labels: List[str]
    disease_label_set: set[str]


class AgeAdapter(torch.nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class SpecializedOriginHead(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 32),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class ContextHead(torch.nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImmuneContextSeparator(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


def _topk_neighbors(query_emb: np.ndarray, ref_emb: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    sims = query_emb @ ref_emb.T
    k = min(top_k, sims.shape[1])
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    row = np.arange(query_emb.shape[0])[:, None]
    scores = sims[row, idx]
    order = np.argsort(-scores, axis=1)
    return idx[row, order], scores[row, order]


def _weighted_vote(labels: Iterable[str], scores: Iterable[float], allowed: set[str] | None = None) -> Tuple[str, float, Dict[str, float]]:
    votes: Dict[str, float] = {}
    for label, score in zip(labels, scores):
        label = str(label)
        if label in {"unknown", "nan", "None", ""}:
            continue
        if allowed is not None and label not in allowed:
            continue
        votes[label] = votes.get(label, 0.0) + float(max(score, 0.0) + 1e-6)
    if not votes:
        return "unknown", 0.0, {}
    total = max(sum(votes.values()), 1e-6)
    best = max(votes, key=votes.get)
    dist = {k: float(v / total) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
    return best, float(votes[best] / total), dist


def _weighted_knn_regression(values: np.ndarray, scores: np.ndarray) -> Tuple[float | None, float]:
    vals = np.asarray(values, dtype=np.float32)
    mask = ~np.isnan(vals)
    if not mask.any():
        return None, 0.0
    w = np.clip(np.asarray(scores, dtype=np.float32)[mask], 0, None) + 1e-6
    return float(np.average(vals[mask], weights=w)), float(np.mean(scores[mask]))


def _confidence_band(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "moderate"
    return "low"


def _build_uncertainty_report(
    coverage: float,
    age_conf: float,
    sex_head_conf: float,
    tumor_prob: float,
    site_conf: float,
    disease_conf: float,
    disease_family_conf: float,
) -> dict:
    tumor_conf = float(max(tumor_prob, 1.0 - tumor_prob))
    fields = {
        "input_gene_coverage": {"score": float(coverage), "band": _confidence_band(float(coverage))},
        "age": {"score": float(age_conf), "band": _confidence_band(float(age_conf))},
        "sex": {"score": float(sex_head_conf), "band": _confidence_band(float(sex_head_conf))},
        "tumor_status": {"score": tumor_conf, "band": _confidence_band(tumor_conf)},
        "site": {"score": float(site_conf), "band": _confidence_band(float(site_conf))},
        "disease_label": {"score": float(disease_conf), "band": _confidence_band(float(disease_conf))},
        "disease_family": {"score": float(disease_family_conf), "band": _confidence_band(float(disease_family_conf))},
    }
    overall = float(np.mean([v["score"] for v in fields.values()]))
    weak = [k for k, v in fields.items() if v["band"] == "low"]
    return {
        "overall_score": overall,
        "overall_band": _confidence_band(overall),
        "weak_fields": weak,
        "fields": fields,
    }


def _age_band(age_years: float | None) -> str:
    if age_years is None or np.isnan(age_years):
        return "unknown"
    if age_years < 45:
        return "younger-adult"
    if age_years < 70:
        return "midlife-adult"
    return "older-adult"


def _withhold_reason(field: str, conf: float, threshold: float, high_confidence_only: bool, overall_band: str) -> str:
    if high_confidence_only and overall_band != "high":
        return "suppressed by high-confidence-only mode"
    if conf < threshold:
        return f"insufficient evidence ({conf:.2f} < {threshold:.2f})"
    return "reported"


def _build_report_card(
    *,
    uncertainty: dict,
    report_age_years: float | None,
    report_age_method: str,
    report_age_conf: float,
    sex_label: str,
    sex_conf: float,
    tumor_label: str,
    tumor_conf: float,
    report_tumor_status: str,
    report_tumor_mode: str,
    report_site: str,
    site_conf: float,
    report_disease_family: str,
    disease_family_conf: float,
    report_disease_label: str,
    report_canonical_disease_label: str,
    disease_conf: float,
    thoracic_origin_prob: float,
    context_label: str,
    context_conf: float,
    high_confidence_only: bool,
) -> dict:
    overall_band = str(uncertainty.get("overall_band", "low"))
    safe_calls: dict[str, dict] = {}
    context_calls: dict[str, dict] = {}
    withheld_calls: dict[str, dict] = {}

    if report_age_years is not None and report_age_conf >= 0.80 and (not high_confidence_only or overall_band == "high"):
        safe_calls["age_exact"] = {
            "value": float(report_age_years),
            "method": report_age_method,
            "confidence": float(report_age_conf),
        }
    elif report_age_years is not None and report_age_conf >= 0.50:
        context_calls["age_band"] = {
            "value": _age_band(report_age_years),
            "method": report_age_method,
            "confidence": float(report_age_conf),
            "reason": "exact age not stable enough; reporting broader age band",
        }
    else:
        withheld_calls["age"] = {
            "reason": _withhold_reason("age", float(report_age_conf), 0.50, high_confidence_only, overall_band)
        }

    safe_calls["sex"] = {"value": sex_label, "confidence": float(sex_conf)}
    if report_tumor_mode == "exact":
        safe_calls["tumor_status"] = {"value": report_tumor_status, "confidence": float(tumor_conf)}
    elif report_tumor_mode == "context":
        context_calls["tumor_status_context"] = {
            "value": report_tumor_status,
            "confidence": float(tumor_conf),
            "reason": _withhold_reason("tumor_status", float(tumor_conf), 0.75, high_confidence_only, overall_band),
            "fallback": report_tumor_status,
        }
    else:
        withheld_calls["tumor_status"] = {
            "reason": _withhold_reason("tumor_status", float(tumor_conf), 0.75, high_confidence_only, overall_band)
        }

    if "context" not in report_site and report_site != "unknown":
        safe_calls["site"] = {"value": report_site, "confidence": float(site_conf)}
    else:
        context_calls["site_context"] = {
            "value": report_site,
            "confidence": float(site_conf),
            "reason": _withhold_reason("site", float(site_conf), 0.65, high_confidence_only, overall_band),
            "fallback": report_site,
        }

    if "suppressed" not in report_disease_family and "uncertain" not in report_disease_family:
        safe_calls["disease_family"] = {
            "value": report_disease_family,
            "confidence": float(disease_family_conf),
        }
    else:
        context_calls["disease_family_context"] = {
            "value": report_disease_family,
            "confidence": float(disease_family_conf),
            "reason": _withhold_reason(
                "disease_family", float(disease_family_conf), 0.65, high_confidence_only, overall_band
            ),
            "fallback": report_disease_family,
        }

    if "suppressed" not in report_disease_label and "not stable enough" not in report_disease_label:
        safe_calls["disease_label"] = {
            "value": report_disease_label,
            "confidence": float(disease_conf),
        }
    else:
        withheld_calls["disease_label"] = {
            "reason": _withhold_reason(
                "disease_label", float(disease_conf), 0.55, high_confidence_only, overall_band
            ),
            "fallback": report_disease_label,
        }

    if "suppressed" not in report_canonical_disease_label and "not stable enough" not in report_canonical_disease_label:
        safe_calls["canonical_disease_label"] = {
            "value": report_canonical_disease_label,
            "confidence": float(disease_conf),
        }
    else:
        withheld_calls["canonical_disease_label"] = {
            "reason": _withhold_reason(
                "canonical_disease_label", float(disease_conf), 0.55, high_confidence_only, overall_band
            ),
            "fallback": report_canonical_disease_label,
        }

    if thoracic_origin_prob >= 0.75:
        safe_calls["thoracic_origin_support"] = {
            "value": "thoracic origin supported",
            "confidence": float(thoracic_origin_prob),
        }
    elif thoracic_origin_prob >= 0.35:
        context_calls["thoracic_origin_support"] = {
            "value": "possible thoracic origin",
            "confidence": float(thoracic_origin_prob),
            "reason": "thoracic-origin head provides partial support but not enough for an exact origin call",
            "fallback": "possible thoracic origin",
        }

    if context_label != "unknown":
        if context_conf >= 0.75:
            safe_calls["context_state"] = {
                "value": context_label.replace("_", " "),
                "confidence": float(context_conf),
            }
        elif context_conf >= 0.50:
            context_calls["context_state"] = {
                "value": context_label.replace("_", " "),
                "confidence": float(context_conf),
                "reason": "context discriminator provides partial but not decisive support",
                "fallback": context_label.replace("_", " "),
            }

    return {
        "evidence_strength": {
            "overall_band": overall_band,
            "overall_score": float(uncertainty.get("overall_score", 0.0)),
        },
        "safe_calls": safe_calls,
        "context_calls": context_calls,
        "withheld_calls": withheld_calls,
    }


def _insufficient_input_response(
    *,
    raw: pd.Series,
    resolved_mode: str,
    matched_genes: int,
    coverage: float,
    reason: str,
) -> dict:
    withheld = {
        "sex": {"reason": reason},
        "tumor_status": {"reason": reason},
        "site": {"reason": reason},
        "disease_family": {"reason": reason},
        "canonical_disease_label": {"reason": reason},
        "disease_label": {"reason": reason},
        "age": {"reason": reason},
    }
    support_profile = {
        "support_level": "unsupported",
        "support_confidence": 1.0,
        "reasons": [reason],
    }
    input_payload = {
        "n_rows_parsed": int(len(raw)),
        "n_unique_genes": int(raw.index.nunique()),
        "n_selected_genes_matched": int(matched_genes),
        "selected_gene_coverage": float(coverage),
        "input_quality_tier": "insufficient",
        "value_mode": resolved_mode,
        "input_sufficient": False,
        "insufficiency_reason": reason,
        "top_archetype": "unknown",
        "top_archetype_score": 0.0,
    }
    predictions = {
        "report_age_years": None,
        "report_age_method": "withheld",
        "report_age_conf": 0.0,
        "report_age_band": "not stable enough to report",
        "report_age_mode": "withheld",
        "pred_sex_head": "withheld",
        "pred_sex_head_conf": 0.0,
        "pred_tumor_status_head": "withheld",
        "pred_tumor_prob_head": 0.0,
        "pred_anatomical_site": "withheld",
        "pred_site_conf": 0.0,
        "pred_disease_label": "withheld",
        "pred_canonical_disease_label": "withheld",
        "pred_disease_conf": 0.0,
        "pred_disease_family": "withheld",
        "pred_disease_family_conf": 0.0,
        "pred_organ_family": "withheld",
        "report_site": "not enough matched genes to infer a stable tissue context",
        "report_site_mode": "withheld",
        "report_disease_family": "not stable enough to report",
        "report_disease_family_mode": "withheld",
        "report_disease_label": "not stable enough to report",
        "report_disease_label_mode": "withheld",
        "report_canonical_disease_label": "not stable enough to report",
        "report_canonical_disease_label_mode": "withheld",
        "pred_support_level": "unsupported",
        "pred_support_confidence": 1.0,
        "pred_support_reasons": [reason],
        "pred_context_label": "unsupported",
        "pred_context_conf": 1.0,
        "pred_context_probs": {},
        "pred_context_origin_fusion": {"fusion_label": "unsupported", "fusion_conf": 1.0},
        "origin_hypothesis": "none",
        "origin_hypothesis_kind": "none",
        "pred_thoracic_origin_prob": 0.0,
    }
    uncertainty = {
        "overall_band": "low",
        "overall_score": 0.0,
        "weak_fields": ["input_gene_coverage", "sex", "tumor_status", "site", "disease_family", "disease_label", "age"],
        "fields": {"input_gene_coverage": {"score": float(coverage), "band": "low"}},
    }
    summary_text = (
        f"Insufficient evidence to profile this RNA sample: only {matched_genes} selected genes matched "
        f"({coverage:.3%} coverage). Provide a fuller expression profile before reporting phenotype."
    )
    return {
        "input": input_payload,
        "predictions": predictions,
        "distributions": {"site": {}, "disease_label": {}, "disease_family": {}, "sex_knn": {}},
        "nearest_neighbors": [],
        "uncertainty": uncertainty,
        "report_card": {
            "evidence_strength": {"overall_band": "low", "overall_score": 0.0},
            "safe_calls": {},
            "context_calls": {},
            "withheld_calls": {
                **withheld,
                "profile_support": {"reason": reason},
            },
        },
        "summary_text": summary_text,
        "explainer": _build_openworld_explainer(
            input_payload=input_payload,
            predictions=predictions,
            uncertainty=uncertainty,
            archetype_scores={},
            relative_signature_scores={},
            support=[],
            support_profile=support_profile,
            summary_text=summary_text,
        ),
    }


def _parse_gene_count_lines(text: str) -> pd.Series:
    rows: List[Tuple[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
        elif "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        else:
            parts = line.split()
        if len(parts) < 2:
            continue
        gene = parts[0].upper()
        gene = gene.split(".")[0]
        try:
            value = float(parts[1])
        except ValueError:
            continue
        rows.append((gene, value))
    if not rows:
        raise ValueError("No valid gene,count rows were parsed.")
    df = pd.DataFrame(rows, columns=["gene", "value"])
    series = df.groupby("gene", as_index=True)["value"].sum()
    return series.astype(np.float32)


def _transform_input(values: pd.Series, value_mode: str) -> Tuple[pd.Series, str]:
    mode = value_mode
    if mode == "auto":
        frac_nonint = float(np.mean(np.abs(values.to_numpy() - np.round(values.to_numpy())) > 1e-6))
        vmax = float(values.max()) if len(values) else 0.0
        if vmax > 50 or frac_nonint < 0.1:
            mode = "raw_count"
        else:
            mode = "log1p"
    if mode == "raw_count":
        return np.log1p(values.clip(lower=0)).astype(np.float32), mode
    if mode == "log1p":
        return values.astype(np.float32), mode
    raise ValueError(f"Unsupported value_mode: {value_mode}")


def _to_logcpm_text(raw: pd.Series) -> str:
    vals = pd.to_numeric(raw, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(vals.sum())
    if total <= 0:
        return ""
    cpm = (vals / total) * 1e6
    logcpm = np.log1p(cpm)
    return "\n".join(f"{gene},{float(val):.6f}" for gene, val in logcpm.items())


def _compute_archetype_scores(values: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0)
    scores: dict[str, float] = {}
    for name, genes in ARCHETYPE_MARKERS.items():
        present = [g for g in genes if g in vals.index]
        if not present:
            scores[name] = 0.0
            continue
        scores[name] = float(vals.loc[present].mean())
    return scores


def _compute_relative_signature_scores(values: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(np.float32)
    arr = vals.to_numpy(dtype=np.float32)
    median = float(np.median(arr)) if len(arr) else 0.0
    q1 = float(np.quantile(arr, 0.25)) if len(arr) else 0.0
    q3 = float(np.quantile(arr, 0.75)) if len(arr) else 1.0
    iqr = max(q3 - q1, 1e-3)
    scores: dict[str, float] = {}
    for name, genes in RELATIVE_SIGNATURE_MARKERS.items():
        present = [g for g in genes if g in vals.index]
        if not present:
            scores[name] = 0.0
            continue
        rel = (vals.loc[present].astype(np.float32) - median) / iqr
        scores[name] = float(rel.mean())
    return scores


def _specialized_origin_features(
    emb_1d: np.ndarray,
    archetype_scores: dict[str, float],
    tumor_prob: float,
) -> np.ndarray:
    extra = np.asarray([float(archetype_scores.get(k, 0.0)) for k in ARCHETYPE_MARKERS.keys()] + [float(tumor_prob)], dtype=np.float32)
    return np.concatenate([np.asarray(emb_1d, dtype=np.float32), extra], axis=0)


def _archetype_context(top_archetype: str, organ_family_label: str) -> str:
    return ARCHETYPE_CONTEXT_LABELS.get(top_archetype, f"{organ_family_label} tissue context")


def _synthesize_explainer_tissue_context(
    *,
    context_state: str,
    report_site: str,
    report_site_mode: str,
    report_disease_family: str,
    top_archetype: str,
    organ_family_label: str,
    fusion_label: str,
    rescue_label: str,
    thoracic_support: float,
    input_quality_tier: str,
) -> str:
    site = str(report_site or "unknown")
    site_lower = site.lower()
    family = str(organ_family_label or "other")
    rescue = str(rescue_label or "none")
    context_state = str(context_state or "unknown")
    disease_family_text = str(report_disease_family or "").lower()

    if input_quality_tier == "insufficient":
        return "not enough matched genes to infer a stable tissue context"

    if context_state == "clean_context" and report_site_mode == "exact":
        if family == "other" and str(report_disease_family).lower() in {
            "other_non_tumor",
            "healthy_control",
            "infectious",
            "autoimmune_inflammatory",
        } and rescue == "none":
            return "other tissue context"
        if family == "cutaneous" and str(report_disease_family).lower() == "cutaneous_non_tumor" and site_lower in {"liver", "brain", "stomach", "pancreas", "colon"}:
            return "cutaneous tissue context"
        return site

    if rescue == "hematologic_rescue":
        return "hematologic / immune-dominant context"
    if rescue == "breast_rescue":
        return "breast epithelial context"
    if rescue == "liver_rescue":
        return "liver tissue context"
    if rescue == "brain_rescue":
        return "brain tissue context"

    if fusion_label in {"immune_heavy_tumor_context", "thoracic_infiltrated_tumor_context"}:
        if report_site_mode == "exact" and site_lower in {"blood", "bone_marrow"}:
            return site
        if "immune-dominant" in site_lower:
            return "immune-dominant tissue context"
        if thoracic_support >= 0.75:
            return "thoracic epithelial-immune context"
        if "hematologic" in disease_family_text or report_site in {"blood", "bone_marrow"}:
            return "hematologic / immune-dominant context"
        if report_site_mode == "exact" and site_lower not in {"unknown", "other"}:
            return f"{site} with heavy immune infiltration"
        if family in {"breast", "digestive", "cutaneous", "cns", "genitourinary"}:
            return f"{family} epithelial-immune context"
        return "mixed epithelial-immune context"

    if fusion_label == "activated_non_malignant_immune_context":
        if site_lower in {"blood", "bone_marrow"}:
            return site
        if "hematologic" in disease_family_text or rescue == "hematologic_rescue":
            return "hematologic / immune-dominant context"
        if family in {"breast", "digestive", "cutaneous", "cns", "genitourinary", "thoracic"}:
            return f"{family} immune-dominant context"
        return "immune-dominant tissue context"

    if report_site_mode == "exact":
        if site_lower in {"blood", "bone_marrow"}:
            return "hematologic tissue context"
        return site

    if "mixed tissue-of-origin context" in site_lower:
        return "mixed tissue-of-origin context"
    if "immune-dominant" in site_lower:
        if "hematologic" in disease_family_text or rescue == "hematologic_rescue":
            return "hematologic / immune-dominant context"
        return "immune-dominant tissue context"
    if "hematologic" in site_lower:
        return "hematologic tissue context"
    if "tissue context" in site_lower:
        return site
    if family != "other":
        return f"{family} tissue context"
    return site


def _synthesize_explainer_origin(
    *,
    hypothesis: str,
    kind: str,
    thoracic_support: float,
    tissue_context: str,
    support_level: str,
) -> tuple[str, str]:
    tissue = str(tissue_context or "").lower()
    hypothesis = str(hypothesis or "none")
    kind = str(kind or "none")

    if support_level == "unsupported":
        return "none", "none"
    if hypothesis == "none":
        return "none", "none"
    if "hematologic" in tissue and thoracic_support < 0.35:
        return "none", "none"
    if (
        "immune-dominant" in tissue
        and "hematologic" in tissue
        and kind in {"squamous_epithelial", "epithelial_solid_tumor"}
        and thoracic_support < 0.35
    ):
        return "none", "none"
    if "mixed tissue-of-origin" in tissue and thoracic_support < 0.35:
        return "none", "none"
    return hypothesis, kind


def _synthesize_explainer_context_state(
    *,
    predicted_context_label: str,
    fusion_label: str,
    support_level: str,
    tissue_context: str,
    report_disease_family: str,
    report_tumor_status: str,
    report_tumor_mode: str,
    origin_hypothesis: str,
    thoracic_support: float,
    immune_tumor_separator_prob: float,
) -> tuple[str, float]:
    predicted = str(predicted_context_label or "unknown")
    fusion = str(fusion_label or "unknown")
    tissue = str(tissue_context or "").lower()
    family_text = str(report_disease_family or "").lower()
    tumor_text = str(report_tumor_status or "").lower()
    tumor_mode = str(report_tumor_mode or "unknown")
    origin_hypothesis = str(origin_hypothesis or "none")

    if support_level == "unsupported":
        return "unsupported", 1.0
    if fusion == "thoracic_infiltrated_tumor_context" and thoracic_support >= 0.35:
        return "thoracic_infiltrated_tumor_context", max(0.35, thoracic_support)
    if fusion == "activated_non_malignant_immune_context":
        return "activated_non_malignant_immune_context", 0.6
    if predicted == "immune_heavy_tumor_context" or fusion == "immune_heavy_tumor_context":
        if (
            tumor_mode == "context"
            and origin_hypothesis == "none"
            and thoracic_support < 0.35
            and family_text in {"immune-dominant context", "hematologic context", "healthy_control", "uncertain phenotype family"}
            and ("hematologic" in tissue or "immune-dominant" in tissue or "skin" in tissue or tissue in {"blood", "bone_marrow"})
        ):
            return "activated_non_malignant_immune_context", 0.58
        if (
            ("hematologic" in tissue or tissue in {"blood", "bone_marrow"})
            and tumor_mode == "context"
            and thoracic_support < 0.35
            and immune_tumor_separator_prob < 0.55
        ):
            return "activated_non_malignant_immune_context", 0.55
        if (
            "immune-dominant" in tissue
            and "uncertain or activated non-malignant" in tumor_text
            and thoracic_support < 0.35
            and immune_tumor_separator_prob < 0.55
        ):
            return "activated_non_malignant_immune_context", 0.52
        return "immune_heavy_tumor_context", max(0.35, immune_tumor_separator_prob, thoracic_support)
    if predicted in {"clean_context", "activated_non_malignant_immune_context", "thoracic_infiltrated_tumor_context"}:
        return predicted, 0.5
    return predicted, 0.35 if predicted != "unknown" else 0.0


def _compute_explainer_quality_tier(
    *,
    support_level: str,
    support_confidence: float,
    context_state: str,
    tissue_context: str,
    origin_hypothesis: str,
    thoracic_support: float,
    report_site_mode: str,
    report_tumor_mode: str,
    input_quality_tier: str,
) -> str:
    if input_quality_tier == "insufficient":
        return "low"
    if support_level == "unsupported":
        return "low"
    if (
        support_level == "supported"
        and support_confidence >= 0.65
        and report_site_mode == "exact"
        and report_tumor_mode in {"exact", "context"}
        and context_state in {"clean_context", "thoracic_infiltrated_tumor_context"}
    ):
        return "high"
    if (
        context_state in {"immune_heavy_tumor_context", "activated_non_malignant_immune_context"}
        and (
            "immune-dominant" in str(tissue_context).lower()
            or "hematologic" in str(tissue_context).lower()
            or origin_hypothesis != "none"
            or thoracic_support >= 0.35
        )
    ):
        return "medium"
    if support_level == "mixed_interpretable":
        return "medium"
    return "low"


def _map_explainer_route(
    quality_tier: str,
    *,
    support_level: str,
    support_confidence: float,
    context_state: str,
    tissue_context: str,
    router_score: float | None = None,
) -> tuple[str, str]:
    tier = str(quality_tier or "low")
    tissue = str(tissue_context or "").lower()
    if (
        tier in {"high", "medium"}
        and support_level in {"supported", "mixed_interpretable"}
        and support_confidence >= 0.65
        and context_state in {"clean_context", "activated_non_malignant_immune_context"}
        and any(tok in tissue for tok in ["blood", "hematologic", "immune-dominant"])
        ):
        return (
            "good_explainer",
            "strong enough for phenotype-level interpretation under the current atlas",
        )
    if (
        router_score is not None
        and router_score >= 0.65
        and tier in {"high", "medium"}
        and support_level in {"supported", "mixed_interpretable"}
    ):
        return (
            "mixed_explainer_high_value",
            "not stable enough for full phenotype-level reporting, but strong enough for a high-value context/origin explanation",
        )
    if tier in {"high", "medium"}:
        return (
            "mixed_explainer",
            "better interpreted as context/composition/origin evidence than as a stable phenotype entity",
        )
    return (
        "unsupported_explainer",
        "not supported strongly enough for a reliable open-world explanation under the current atlas",
    )


def _compose_route_verdict(
    *,
    route_label: str,
    summary_text: str,
    tissue_context: str,
    context_state: str,
    origin_hypothesis: str,
    report_disease_family: str,
    report_canonical_disease_label: str,
) -> str:
    route = str(route_label or "unsupported_explainer")
    tissue = str(tissue_context or "unknown")
    context = str(context_state or "unknown").replace("_", " ")
    origin = str(origin_hypothesis or "none")
    family = str(report_disease_family or "unknown").replace("_", " ")
    canonical = str(report_canonical_disease_label or "unknown")

    if route == "good_explainer":
        if canonical not in {"unknown", "not stable enough to report", "suppressed in high-confidence-only mode"}:
            return f"Good explainer route: supported phenotype-like interpretation in {tissue}, most consistent with {canonical}."
        return f"Good explainer route: supported phenotype-like interpretation in {tissue}, with stable {family} context."
    if route == "mixed_explainer_high_value":
        tail = f" Origin hypothesis: {origin}." if origin not in {"none", "unknown"} else ""
        return f"High-value mixed route: the sample is still open-world and partially mixed, but the context/origin evidence is strong enough to support a focused biological explanation in {tissue}.{tail}"
    if route == "mixed_explainer":
        tail = f" Origin hypothesis: {origin}." if origin not in {"none", "unknown"} else ""
        return f"Mixed explainer route: interpret primarily through context and composition. Current context is {context} in {tissue}; phenotype entities should be treated as tentative.{tail}"
    return f"Unsupported explainer route: current atlas support is insufficient for a reliable phenotype-level explanation. Treat this sample primarily as unsupported or OOD. {summary_text}"


def _compute_context_origin_fusion(
    *,
    tumor_prob: float,
    thoracic_origin_prob: float,
    immune_heavy_prob: float,
    activated_non_malignant_prob: float,
    top_archetype: str,
    organ_family_label: str,
) -> dict[str, float | str]:
    epithelial_bonus = 0.10 if top_archetype in {"epithelial_dominant", "squamous_like", "thoracic_like"} else 0.0
    immune_bonus = 0.08 if top_archetype in {"immune_dominant", "hematologic_like"} else 0.0
    thoracic_bonus = 0.08 if organ_family_label == "thoracic" else 0.0

    immune_tumor = float(
        min(
            1.0,
            0.55 * immune_heavy_prob
            + 0.20 * float(max(tumor_prob - 0.45, 0.0))
            + 0.20 * thoracic_origin_prob
            + epithelial_bonus,
        )
    )
    activated_non_malignant = float(
        min(
            1.0,
            0.70 * activated_non_malignant_prob
            + 0.20 * float(max(0.60 - tumor_prob, 0.0))
            + immune_bonus,
        )
    )
    thoracic_infiltrated = float(
        min(
            1.0,
            0.60 * thoracic_origin_prob
            + 0.25 * immune_heavy_prob
            + 0.10 * float(max(tumor_prob - 0.50, 0.0))
            + thoracic_bonus,
        )
    )
    if thoracic_infiltrated >= max(0.55, immune_tumor + 0.12):
        fusion_label = "thoracic_infiltrated_tumor_context"
        fusion_conf = thoracic_infiltrated
    elif immune_tumor >= max(0.35, activated_non_malignant + 0.08):
        fusion_label = "immune_heavy_tumor_context"
        fusion_conf = immune_tumor
    elif activated_non_malignant >= 0.22:
        fusion_label = "activated_non_malignant_immune_context"
        fusion_conf = activated_non_malignant
    else:
        fusion_label = "clean_context"
        fusion_conf = float(max(0.0, 1.0 - max(immune_tumor, activated_non_malignant, thoracic_infiltrated)))
    return {
        "fusion_label": fusion_label,
        "fusion_conf": fusion_conf,
        "immune_heavy_tumor_score": immune_tumor,
        "activated_non_malignant_score": activated_non_malignant,
        "thoracic_infiltrated_score": thoracic_infiltrated,
    }


def _compute_support_profile(
    *,
    input_quality_tier: str,
    report_tumor_status: str,
    report_tumor_mode: str,
    report_site: str,
    report_site_mode: str,
    report_disease_family: str,
    report_disease_family_mode: str,
    report_canonical_disease_label: str,
    report_canonical_disease_label_mode: str,
    pred_context_origin_fusion: dict[str, float | str] | None,
    origin_hypothesis: str,
    pred_topk_gse_n: int,
    pred_site_conf: float,
    pred_disease_family_conf: float,
    pred_disease_conf: float,
    pred_tumor_prob_head: float,
) -> dict[str, object]:
    fusion_label = str((pred_context_origin_fusion or {}).get("fusion_label", "unknown"))
    clean_like = fusion_label in {"clean_context", "thoracic_infiltrated_tumor_context"}
    mixed_like = fusion_label in {"immune_heavy_tumor_context", "activated_non_malignant_immune_context"}
    site_text = str(report_site).lower()
    tumor_text = str(report_tumor_status).lower()

    unsupported_reasons: list[str] = []
    if input_quality_tier in {"insufficient", "sparse"}:
        unsupported_reasons.append("input quality below phenotype-report threshold")
    if "unsupported mixed biological context" in tumor_text:
        unsupported_reasons.append("mixed biological context under alternative routes")
    if "mixed tissue-of-origin context" in site_text:
        unsupported_reasons.append("tissue-of-origin remains mixed")
    if pred_topk_gse_n <= 1 and report_site_mode != "exact" and report_canonical_disease_label_mode != "exact":
        unsupported_reasons.append("nearest-neighbor evidence collapses to a single project")

    if unsupported_reasons:
        return {
            "support_level": "unsupported",
            "support_confidence": float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.35
                        + 0.15 * float(input_quality_tier == "insufficient")
                        + 0.15 * float("mixed tissue-of-origin context" in site_text)
                        + 0.15 * float("unsupported mixed biological context" in tumor_text)
                        + 0.20 * float(pred_topk_gse_n <= 1),
                    ),
                )
            ),
            "reasons": unsupported_reasons,
        }

    supported_reasons: list[str] = []
    exact_site = report_site_mode == "exact"
    exact_tumor = report_tumor_mode == "exact"
    exact_family = report_disease_family_mode == "exact"
    exact_canonical = report_canonical_disease_label_mode == "exact"
    if (
        input_quality_tier in {"adequate", "strong"}
        and exact_site
        and exact_tumor
        and clean_like
        and pred_topk_gse_n >= 2
        and pred_site_conf >= 0.55
        and pred_tumor_prob_head >= 0.60
        and (exact_canonical or (exact_family and pred_disease_family_conf >= 0.55))
    ):
        supported_reasons.append("stable site/tumor evidence with clean or origin-supported context")
        if exact_canonical:
            supported_reasons.append("canonical disease label passes exact reporting gate")
        else:
            supported_reasons.append("disease-family evidence is exact even without canonical entity")
        return {
            "support_level": "supported",
            "support_confidence": float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.30
                        + 0.20 * float(pred_site_conf)
                        + 0.15 * float(pred_tumor_prob_head)
                        + 0.15 * float(pred_disease_family_conf)
                        + 0.20 * float(pred_disease_conf if exact_canonical else pred_disease_family_conf),
                    ),
                )
            ),
            "reasons": supported_reasons,
        }

    mixed_reasons: list[str] = []
    if mixed_like:
        mixed_reasons.append(f"context/origin fusion indicates {fusion_label}")
    if report_site_mode == "context":
        mixed_reasons.append("site is only stable at context level")
    if report_disease_family_mode == "context":
        mixed_reasons.append("disease family is only stable at context level")
    if report_canonical_disease_label_mode != "exact":
        mixed_reasons.append("canonical disease entity is not stable enough to report")
    if origin_hypothesis and origin_hypothesis != "none":
        mixed_reasons.append("origin hypothesis is available but not yet a stable entity-level conclusion")
    if not mixed_reasons:
        mixed_reasons.append("coarse phenotype signals are present, but structured entity-level support is incomplete")
    return {
        "support_level": "mixed_interpretable",
        "support_confidence": float(
            max(
                0.0,
                min(
                    1.0,
                    0.30
                    + 0.15 * float(pred_site_conf)
                    + 0.15 * float(pred_tumor_prob_head)
                    + 0.10 * float(pred_disease_family_conf)
                    + 0.10 * float(bool(origin_hypothesis and origin_hypothesis != "none"))
                    + 0.10 * float(mixed_like),
                ),
            )
        ),
        "reasons": mixed_reasons,
    }


def _build_openworld_explainer(
    *,
    input_payload: dict,
    predictions: dict,
    uncertainty: dict,
    archetype_scores: dict[str, float],
    relative_signature_scores: dict[str, float],
    support: list[dict],
    support_profile: dict[str, object],
    summary_text: str,
) -> dict[str, object]:
    sorted_archetypes = sorted(archetype_scores.items(), key=lambda kv: -float(kv[1]))
    sorted_relative = sorted(relative_signature_scores.items(), key=lambda kv: -float(kv[1]))
    dominant_programs = [
        {
            "name": name,
            "score": float(score),
        }
        for name, score in sorted_archetypes[:3]
        if float(score) > 0
    ]
    relative_biases = [
        {
            "name": name,
            "score": float(score),
        }
        for name, score in sorted_relative[:3]
        if float(score) > 0
    ]
    evidence_neighbors = [
        {
            "sample_id": str(row.get("sample_id", "unknown")),
            "gse": str(row.get("gse", "unknown")),
            "score": float(row.get("score", 0.0)),
            "site": str(row.get("site", "unknown")),
            "tumor_status": str(row.get("tumor_status", "unknown")),
            "disease_family": str(row.get("disease_family", "unknown")),
        }
        for row in support[:5]
    ]
    context_probs = predictions.get("pred_context_probs", {}) or {}
    context_sorted = sorted(context_probs.items(), key=lambda kv: -float(kv[1]))
    context_evidence = [
        {
            "label": str(name),
            "prob": float(prob),
        }
        for name, prob in context_sorted[:3]
    ]
    support_level = str(support_profile.get("support_level", "unknown"))
    fusion = predictions.get("pred_context_origin_fusion") or {}
    tissue_context = _synthesize_explainer_tissue_context(
        context_state=str(predictions.get("pred_context_label", "unknown")),
        report_site=str(predictions.get("report_site", "unknown")),
        report_site_mode=str(predictions.get("report_site_mode", "unknown")),
        report_disease_family=str(predictions.get("report_disease_family", "unknown")),
        top_archetype=str(input_payload.get("top_archetype", "unknown")),
        organ_family_label=str(predictions.get("pred_organ_family", "other")),
        fusion_label=str(fusion.get("fusion_label", "unknown")),
        rescue_label=str((predictions.get("pred_relative_signature_rescue") or {}).get("rescue_label", "none")),
        thoracic_support=float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
        input_quality_tier=str(input_payload.get("input_quality_tier", "unknown")),
    )
    origin_hypothesis, origin_kind = _synthesize_explainer_origin(
        hypothesis=str(predictions.get("origin_hypothesis", "none")),
        kind=str(predictions.get("origin_hypothesis_kind", "none")),
        thoracic_support=float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
        tissue_context=tissue_context,
        support_level=support_level,
    )
    context_state, context_state_conf = _synthesize_explainer_context_state(
        predicted_context_label=str(predictions.get("pred_context_label", "unknown")),
        fusion_label=str(fusion.get("fusion_label", "unknown")),
        support_level=support_level,
        tissue_context=tissue_context,
        report_disease_family=str(predictions.get("report_disease_family", "unknown")),
        report_tumor_status=str(predictions.get("report_tumor_status", "unknown")),
        report_tumor_mode=str(predictions.get("report_tumor_mode", "unknown")),
        origin_hypothesis=origin_hypothesis,
        thoracic_support=float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
        immune_tumor_separator_prob=float(predictions.get("pred_immune_tumor_separator_prob", 0.0) or 0.0),
    )
    quality_tier = _compute_explainer_quality_tier(
        support_level=support_level,
        support_confidence=float(support_profile.get("support_confidence", 0.0) or 0.0),
        context_state=context_state,
        tissue_context=tissue_context,
        origin_hypothesis=origin_hypothesis,
        thoracic_support=float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
        report_site_mode=str(predictions.get("report_site_mode", "unknown")),
        report_tumor_mode=str(predictions.get("report_tumor_mode", "unknown")),
        input_quality_tier=str(input_payload.get("input_quality_tier", "unknown")),
    )
    router_score = _predict_explainer_router_score(
        support_level=support_level,
        quality_tier=quality_tier,
        context_state=context_state,
        tissue_context=tissue_context,
        origin_hypothesis=origin_hypothesis,
        support_confidence=float(support_profile.get("support_confidence", 0.0) or 0.0),
        thoracic_support=float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
    )
    route_label, route_guidance = _map_explainer_route(
        quality_tier,
        support_level=support_level,
        support_confidence=float(support_profile.get("support_confidence", 0.0) or 0.0),
        context_state=context_state,
        tissue_context=tissue_context,
        router_score=router_score,
    )
    route_verdict = _compose_route_verdict(
        route_label=route_label,
        summary_text=str(summary_text),
        tissue_context=tissue_context,
        context_state=context_state,
        origin_hypothesis=origin_hypothesis,
        report_disease_family=str(predictions.get("report_disease_family", "unknown")),
        report_canonical_disease_label=str(predictions.get("report_canonical_disease_label", "unknown")),
    )
    return {
        "verdict": route_verdict,
        "support": {
            "level": support_level,
            "confidence": float(support_profile.get("support_confidence", 0.0) or 0.0),
            "reasons": [str(x) for x in support_profile.get("reasons", [])],
            "input_quality_tier": str(input_payload.get("input_quality_tier", "unknown")),
            "explainer_quality_tier": quality_tier,
            "route": route_label,
            "route_guidance": route_guidance,
            "router_score": None if router_score is None else float(router_score),
        },
        "context": {
            "state": context_state,
            "state_confidence": context_state_conf,
            "fusion_label": str(fusion.get("fusion_label", "unknown")),
            "fusion_confidence": float(fusion.get("fusion_conf", 0.0) or 0.0),
            "tissue_context": tissue_context,
            "tumor_context": str(predictions.get("report_tumor_status", "unknown")),
            "context_evidence": context_evidence,
        },
        "composition": {
            "dominant_programs": dominant_programs,
            "relative_biases": relative_biases,
            "top_archetype": str(input_payload.get("top_archetype", "unknown")),
            "top_archetype_score": float(input_payload.get("top_archetype_score", 0.0) or 0.0),
        },
        "origin": {
            "hypothesis": origin_hypothesis,
            "kind": origin_kind,
            "thoracic_support": float(predictions.get("pred_thoracic_origin_prob", 0.0) or 0.0),
        },
        "phenotype": {
            "sex": {
                "value": str(predictions.get("pred_sex_head", "unknown")),
                "confidence": float(predictions.get("pred_sex_head_conf", 0.0) or 0.0),
            },
            "age_like": {
                "years": predictions.get("report_age_years"),
                "band": str(predictions.get("report_age_band", "not stable enough to report")),
                "mode": str(predictions.get("report_age_mode", "unknown")),
                "confidence": float(predictions.get("report_age_conf", 0.0) or 0.0),
            },
            "site": {
                "value": str(predictions.get("report_site", "unknown")),
                "mode": str(predictions.get("report_site_mode", "unknown")),
                "confidence": float(predictions.get("pred_site_conf", 0.0) or 0.0),
            },
            "disease_family": {
                "value": str(predictions.get("report_disease_family", "unknown")),
                "mode": str(predictions.get("report_disease_family_mode", "unknown")),
                "confidence": float(predictions.get("pred_disease_family_conf", 0.0) or 0.0),
            },
            "canonical_disease": {
                "value": str(predictions.get("report_canonical_disease_label", "unknown")),
                "mode": str(predictions.get("report_canonical_disease_label_mode", "unknown")),
                "confidence": float(predictions.get("pred_disease_conf", 0.0) or 0.0),
            },
        },
        "neighbors": evidence_neighbors,
        "uncertainty": uncertainty,
    }


def _relative_signature_rescue(
    *,
    relative_scores: dict[str, float],
    report_site: str,
    report_disease_family: str,
    tumor_label: str,
) -> dict[str, str | float]:
    ordered = sorted(relative_scores.items(), key=lambda kv: -kv[1])
    top_name, top_score = ordered[0] if ordered else ("unknown", 0.0)
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    gap = float(top_score - second_score)
    rescue = {
        "relative_top_signature": top_name,
        "relative_top_score": float(top_score),
        "relative_second_score": float(second_score),
        "relative_gap": gap,
        "rescue_label": "none",
        "rescue_conf": 0.0,
    }
    if top_name == "breast_like" and top_score >= 2.5 and gap >= 1.5:
        if str(report_site) in {"blood", "bone_marrow"} or str(report_disease_family) == "hematologic_malignancy":
            rescue["rescue_label"] = "breast_rescue"
            rescue["rescue_conf"] = float(min(1.0, 0.15 * top_score + 0.10 * gap))
    elif top_name == "hematologic_like" and top_score >= 1.0 and gap >= 0.4:
        if str(report_site) in {"liver", "breast"} or str(report_disease_family) in {"digestive_tumor", "breast_tumor"}:
            rescue["rescue_label"] = "hematologic_rescue"
            rescue["rescue_conf"] = float(min(1.0, 0.25 * top_score + 0.10 * gap))
    elif top_name == "liver_like" and top_score >= 1.5 and gap >= 0.5:
        if str(report_site) in {"blood", "bone_marrow"} or "tumor" in str(report_disease_family):
            rescue["rescue_label"] = "liver_rescue"
            rescue["rescue_conf"] = float(min(1.0, 0.20 * top_score + 0.10 * gap))
    elif top_name == "brain_like" and top_score >= 1.5 and gap >= 0.5:
        if str(report_site) not in {"brain", "cns"}:
            rescue["rescue_label"] = "brain_rescue"
            rescue["rescue_conf"] = float(min(1.0, 0.20 * top_score + 0.10 * gap))
    return rescue


def _infer_origin_hypothesis(
    *,
    top_archetype: str,
    archetype_scores: dict[str, float],
    tumor_conf: float,
    thoracic_origin_prob: float,
    report_site: str,
    report_disease_family: str,
    report_canonical_disease_label_mode: str,
) -> tuple[str, str]:
    immune_like = top_archetype in {"immune_dominant", "hematologic_like"}
    if not immune_like:
        return "none", "none"
    if tumor_conf < 0.55:
        return "none", "none"
    if report_canonical_disease_label_mode == "exact":
        return "none", "none"
    if report_disease_family == "healthy_control":
        if archetype_scores.get("squamous_like", 0.0) < 5.0 and archetype_scores.get("thoracic_like", 0.0) < 5.0:
            return "none", "none"
    if str(report_site) not in {"immune-dominant tissue context", "blood"} and "context" not in str(report_site):
        return "none", "none"

    squamous = float(archetype_scores.get("squamous_like", 0.0))
    thoracic = float(archetype_scores.get("thoracic_like", 0.0))
    epithelial = float(archetype_scores.get("epithelial_dominant", 0.0))

    if thoracic_origin_prob >= 0.75:
        return "thoracic tumor origin supported despite heavy immune infiltration", "thoracic_head_supported"
    if squamous >= 5.0:
        return "possible squamous epithelial tumor origin with heavy immune infiltration", "squamous_epithelial"
    if thoracic_origin_prob >= 0.35 and (thoracic >= 4.5 or thoracic_origin_prob >= 0.60):
        return "possible thoracic epithelial tumor origin with heavy immune infiltration", "thoracic_epithelial"
    if epithelial >= 6.5 and report_disease_family != "healthy_control":
        return "possible epithelial solid-tumor origin with heavy immune infiltration", "epithelial_solid_tumor"
    return "none", "none"


@lru_cache(maxsize=1)
def load_specialized_origin_heads() -> dict[str, tuple[torch.nn.Module, float]]:
    outdir = OUTPUT_ROOT / "train_specialized_origin_heads_20260416"
    heads: dict[str, tuple[torch.nn.Module, float]] = {}
    thor_payload_path = outdir / "thoracic_head.pt"
    if thor_payload_path.exists():
        payload = torch.load(thor_payload_path, map_location="cpu", weights_only=False)
        model = SpecializedOriginHead(int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        heads["thoracic"] = (model, float(payload.get("threshold", 0.5)))
    return heads


@lru_cache(maxsize=1)
def load_context_head() -> tuple[torch.nn.Module, dict[str, int]] | None:
    payload_path = OUTPUT_ROOT / "train_context_discriminator_20260417" / "context_head.pt"
    if not payload_path.exists():
        return None
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    vocab = dict(payload["context_vocab"])
    model = ContextHead(int(payload["input_dim"]), len(vocab))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, vocab


@lru_cache(maxsize=1)
def load_immune_context_separator() -> tuple[torch.nn.Module, float] | None:
    payload_path = OUTPUT_ROOT / "train_immune_context_separator_20260417" / "immune_context_separator.pt"
    if not payload_path.exists():
        return None
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    model = ImmuneContextSeparator(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, float(payload.get("threshold", 0.5))


def _resolve_packaged_base_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    for fallback_name in ["rna_language_alignment", "semantic_alignment_backbone"]:
        fallback = OUTPUT_ROOT / fallback_name / "bulk_multimodal_embedding.pt"
        if fallback.exists():
            return fallback
    return p


def _resolve_packaged_base_train_script(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    fallback = MODULE_DIR / "train_rna_language_alignment.py"
    if fallback.exists():
        return fallback
    return p


@lru_cache(maxsize=1)
def load_artifacts(age_adapter_run: str = DEFAULT_AGE_ADAPTER_RUN) -> UnknownSampleArtifacts:
    adapter_payload = torch.load(
        OUTPUT_ROOT / age_adapter_run / "age_adapter.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_path = _resolve_packaged_base_checkpoint(str(adapter_payload["base_checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_script = _resolve_packaged_base_train_script(str(adapter_payload["base_train_script"]))
    train_module = load_module(train_script, "unknown_rna_base_train")

    seed = int(checkpoint.get("config", {}).get("seed", getattr(train_module, "SEED", 42)))
    if hasattr(train_module, "SEED"):
        train_module.SEED = seed
    train_module.set_seed(seed)

    model = train_module.BulkRNALanguageAligner(
        n_genes=len(checkpoint["selected_genes"]),
        n_text_features=checkpoint["config"]["n_text_features"],
        n_sources=len(checkpoint["source_map"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    adapter = AgeAdapter(
        embed_dim=model.age_head.weight.shape[1],
        hidden_dim=int(adapter_payload["adapter_hidden_dim"]),
        dropout=float(adapter_payload["adapter_dropout"]),
    )
    adapter.load_state_dict(adapter_payload["adapter_state_dict"])
    adapter.eval()

    bundle = train_module.load_dataset()
    meta = bundle.meta.copy()
    split_assignments = checkpoint.get("split_assignments", {})
    split_assignments = {str(k): str(v) for k, v in split_assignments.items()}
    if split_assignments:
        meta["split"] = meta["sample_id"].astype(str).map(split_assignments).fillna("unknown")
        train_meta = meta.loc[meta["split"] == "train"].copy().reset_index(drop=True)
    else:
        train_meta, _, _ = train_module.build_splits(meta.copy())
        train_meta = train_meta.reset_index(drop=True)
        meta["split"] = "unknown"
        meta.loc[meta["sample_id"].astype(str).isin(train_meta["sample_id"].astype(str)), "split"] = "train"
    if train_meta.empty:
        train_meta = meta.copy().reset_index(drop=True)
        meta["split"] = "train"
    train_sample_ids = train_meta["sample_id"].astype(str).tolist()
    selected_genes = list(checkpoint["selected_genes"])
    expr = train_module.load_expr_subset(bundle, selected_genes, train_sample_ids)
    x = expr.T.to_numpy(dtype=np.float32)
    expr_mean = np.asarray(adapter_payload["expr_mean"], dtype=np.float32)
    expr_std = np.asarray(adapter_payload["expr_std"], dtype=np.float32)
    expr_std = np.where(expr_std < 1e-3, 1.0, expr_std)
    x = np.clip((x - expr_mean) / expr_std, -8.0, 8.0)

    with torch.no_grad():
        train_emb = model.encode_expr(torch.tensor(x, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    disease_counts = train_meta["feat_disease_label"].astype(str).value_counts()
    disease_labels = [
        label
        for label, count in disease_counts.items()
        if count >= 25 and label not in {"unknown", "nan", "None"}
    ]
    return UnknownSampleArtifacts(
        model=model,
        age_adapter=adapter,
        train_module=train_module,
        checkpoint=checkpoint,
        selected_genes=selected_genes,
        expr_mean=expr_mean,
        expr_std=expr_std,
        train_emb=train_emb,
        train_meta=train_meta,
        disease_labels=disease_labels,
        disease_label_set=set(disease_labels),
    )


def run_unknown_sample(
    counts_text: str,
    top_k: int = DEFAULT_TOP_K,
    value_mode: str = "auto",
    high_confidence_only: bool = False,
) -> dict:
    art = load_artifacts()
    raw = _parse_gene_count_lines(counts_text)
    transformed, resolved_mode = _transform_input(raw, value_mode)
    archetype_scores = _compute_archetype_scores(transformed)
    relative_signature_scores = _compute_relative_signature_scores(transformed)
    archetype_sorted = sorted(archetype_scores.items(), key=lambda kv: -kv[1])
    top_archetype = archetype_sorted[0][0] if archetype_sorted else "unknown"
    top_archetype_score = float(archetype_sorted[0][1]) if archetype_sorted else 0.0
    second_archetype_score = float(archetype_sorted[1][1]) if len(archetype_sorted) > 1 else 0.0
    selected = pd.Series(0.0, index=art.selected_genes, dtype=np.float32)
    overlap = transformed.index.intersection(selected.index)
    selected.loc[overlap] = transformed.loc[overlap].astype(np.float32)
    coverage = float(len(overlap) / max(len(art.selected_genes), 1))
    input_quality_tier = _input_quality_tier(int(len(overlap)), coverage)
    if len(overlap) < MIN_SELECTED_GENES_MATCHED or coverage < MIN_SELECTED_GENE_COVERAGE:
        return _insufficient_input_response(
            raw=raw,
            resolved_mode=resolved_mode,
            matched_genes=int(len(overlap)),
            coverage=coverage,
            reason=(
                f"input coverage too low: matched {len(overlap)} selected genes "
                f"({coverage:.3%}); require at least {MIN_SELECTED_GENES_MATCHED} genes "
                f"and {MIN_SELECTED_GENE_COVERAGE:.1%} coverage"
            ),
        )

    x = selected.to_numpy(dtype=np.float32)
    x = np.clip((x - art.expr_mean) / art.expr_std, -8.0, 8.0)[None, :]
    with torch.no_grad():
        xb = torch.tensor(x, dtype=torch.float32)
        emb = art.model.encode_expr(xb).cpu().numpy().astype(np.float32)
        sex_prob = torch.softmax(art.model.sex_head(torch.tensor(emb, dtype=torch.float32)), dim=1).cpu().numpy()[0]
        tumor_prob_head = torch.sigmoid(art.model.tumor_head(torch.tensor(emb, dtype=torch.float32)).squeeze(1)).cpu().numpy()[0]
        age_z = art.age_adapter(torch.tensor(emb, dtype=torch.float32)).cpu().numpy()[0]
    age_years = float(age_z * float(art.checkpoint["age_std"]) + float(art.checkpoint["age_mean"]))

    idx, scores = _topk_neighbors(emb, art.train_emb, top_k=top_k)
    idx = idx[0]
    scores = scores[0]
    nbr = art.train_meta.iloc[idx].copy().reset_index(drop=True)

    sex_label = "female" if sex_prob[1] >= sex_prob[0] else "male"
    tumor_label = "tumor" if tumor_prob_head >= 0.5 else "non_tumor"
    tumor_conf = float(max(tumor_prob_head, 1.0 - tumor_prob_head))

    site_label, site_conf, site_dist = _weighted_vote(nbr["feat_anatomical_site"], scores)
    disease_label, disease_conf, disease_dist = _weighted_vote(
        nbr["feat_disease_label"],
        scores,
        allowed=art.disease_label_set,
    )
    sex_knn_label, sex_knn_conf, sex_knn_dist = _weighted_vote(nbr["feat_sex"], scores, allowed={"male", "female"})
    age_knn, age_knn_conf = _weighted_knn_regression(pd.to_numeric(nbr["feat_age"], errors="coerce").to_numpy(), scores)

    family_labels = [
        disease_family(dl, st, ts)
        for dl, st, ts in zip(nbr["feat_disease_label"], nbr["feat_anatomical_site"], nbr["feat_tumor_status"])
    ]
    disease_family_label, disease_family_conf, disease_family_dist = _weighted_vote(family_labels, scores)

    support = []
    for row, score in zip(nbr.to_dict(orient="records"), scores):
        support.append(
            {
                "sample_id": str(row["sample_id"]),
                "gse": str(row["gse"]),
                "score": float(score),
                "sex": str(row.get("feat_sex", "unknown")),
                "site": str(row.get("feat_anatomical_site", "unknown")),
                "tumor_status": str(row.get("feat_tumor_status", "unknown")),
                "disease_label": str(row.get("feat_disease_label", "unknown")),
                "disease_family": disease_family(
                    row.get("feat_disease_label"),
                    row.get("feat_anatomical_site"),
                    row.get("feat_tumor_status"),
                ),
                "age": None if pd.isna(row.get("feat_age")) else float(row["feat_age"]),
            }
        )
    pred_canonical_disease_label = normalize_disease_label(disease_label, "ontology")
    matching_canonical_support = [
        row
        for row in support
        if normalize_disease_label(row.get("disease_label", "unknown"), "ontology") == pred_canonical_disease_label
    ]
    support_match_n = int(len(matching_canonical_support))
    support_gse_n = int(len({str(row.get("gse", "unknown")) for row in matching_canonical_support}))
    topk_gse_n = int(len({str(row.get("gse", "unknown")) for row in support}))

    organ_family_label = organ_family(site_label)
    report_age_years = age_knn if age_knn is not None else age_years
    report_age_method = "knn" if age_knn is not None else "adapter"
    report_age_conf = age_knn_conf if age_knn is not None else 0.35
    report_age_band = _age_band(report_age_years) if report_age_conf >= 0.50 else "not stable enough to report"
    uncertainty = _build_uncertainty_report(
        coverage=coverage,
        age_conf=float(report_age_conf),
        sex_head_conf=float(max(sex_prob)),
        tumor_prob=float(tumor_prob_head),
        site_conf=site_conf,
        disease_conf=disease_conf,
        disease_family_conf=disease_family_conf,
    )
    if uncertainty["overall_band"] == "high":
        lead = "Strong evidence suggests"
    elif uncertainty["overall_band"] == "moderate":
        lead = "Moderate evidence suggests"
    else:
        lead = "Weak evidence suggests"
    report_site = site_label
    report_site_mode = "exact"
    if site_conf < 0.65:
        report_site = f"{organ_family_label} tissue context"
        report_site_mode = "context"
    report_disease_family = disease_family_label
    report_disease_family_mode = "exact"
    if disease_family_conf < 0.65:
        report_disease_family = "uncertain phenotype family"
        report_disease_family_mode = "context"
    report_disease_label = disease_label
    report_disease_label_mode = "exact"
    report_tumor_status = tumor_label
    report_tumor_mode = "exact"
    specialized_heads = load_specialized_origin_heads()
    thoracic_origin_prob = 0.0
    if "thoracic" in specialized_heads:
        thor_model, _thor_thr = specialized_heads["thoracic"]
        aux = _specialized_origin_features(emb[0], archetype_scores, tumor_prob_head)[None, :]
        with torch.no_grad():
            thoracic_origin_prob = float(
                torch.sigmoid(thor_model(torch.tensor(aux, dtype=torch.float32))).cpu().numpy()[0]
            )
    context_label = "unknown"
    context_conf = 0.0
    context_probs: dict[str, float] = {}
    immune_tumor_separator_prob = 0.0
    context_head_bundle = load_context_head()
    context_aux = _specialized_origin_features(emb[0], archetype_scores, tumor_prob_head)[None, :]
    if context_head_bundle is not None:
        context_model, context_vocab = context_head_bundle
        with torch.no_grad():
            context_logits = context_model(torch.tensor(context_aux, dtype=torch.float32))
            context_prob = torch.softmax(context_logits, dim=1).cpu().numpy()[0]
        inv_context_vocab = {int(v): str(k) for k, v in context_vocab.items()}
        context_probs = {inv_context_vocab[i]: float(context_prob[i]) for i in range(len(context_prob))}
        context_idx = int(np.argmax(context_prob))
        context_label = inv_context_vocab.get(context_idx, "unknown")
        context_conf = float(context_prob[context_idx])
        immune_heavy_prob = float(context_probs.get("immune_heavy_tumor_context", 0.0))
        activated_non_malignant_prob = float(context_probs.get("activated_non_malignant_immune_context", 0.0))
        if immune_heavy_prob >= 0.30 and immune_heavy_prob > activated_non_malignant_prob:
            context_label = "immune_heavy_tumor_context"
            context_conf = immune_heavy_prob
        elif activated_non_malignant_prob >= 0.20 and activated_non_malignant_prob > immune_heavy_prob:
            context_label = "activated_non_malignant_immune_context"
            context_conf = activated_non_malignant_prob
    immune_context_separator_bundle = load_immune_context_separator()
    if immune_context_separator_bundle is not None:
        immune_model, _immune_thr = immune_context_separator_bundle
        with torch.no_grad():
            immune_tumor_separator_prob = float(
                torch.sigmoid(immune_model(torch.tensor(context_aux, dtype=torch.float32))).cpu().numpy()[0]
            )
    fusion = _compute_context_origin_fusion(
        tumor_prob=float(tumor_prob_head),
        thoracic_origin_prob=float(thoracic_origin_prob),
        immune_heavy_prob=float(context_probs.get("immune_heavy_tumor_context", 0.0)),
        activated_non_malignant_prob=float(context_probs.get("activated_non_malignant_immune_context", 0.0)),
        top_archetype=top_archetype,
        organ_family_label=organ_family_label,
    )
    rescue = _relative_signature_rescue(
        relative_scores=relative_signature_scores,
        report_site=report_site,
        report_disease_family=report_disease_family,
        tumor_label=tumor_label,
    )
    # Family-first disease reporting: only surface a fine label if the broader
    # disease-family context is also stable enough.
    if disease_family_conf < 0.65 or disease_conf < 0.55:
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
    if high_confidence_only and uncertainty["overall_band"] != "high":
        if site_conf < 0.85:
            report_site = f"{organ_family_label} tissue context"
            report_site_mode = "context"
        if disease_family_conf < 0.85:
            report_disease_family = "suppressed in high-confidence-only mode"
            report_disease_family_mode = "withheld"
        if disease_conf < 0.85:
            report_disease_label = "suppressed in high-confidence-only mode"
            report_disease_label_mode = "withheld"
        if tumor_conf < 0.85:
            report_tumor_status = "uncertain tumor-status context"
            report_tumor_mode = "context"

    # Input-quality-aware reporting: sparse and partial uploads should be
    # treated as coarse phenotype/context inference rather than exact disease
    # calls, even if the local neighborhood looks deceptively confident.
    if input_quality_tier == "sparse":
        report_tumor_status = "uncertain tumor-status context"
        report_tumor_mode = "context"
        report_site = _archetype_context(top_archetype, organ_family_label)
        report_site_mode = "context"
        report_disease_family = "not stable enough to report"
        report_disease_family_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
        if report_age_conf < 0.50:
            report_age_band = "not stable enough to report"
        if report_age_conf < 0.80:
            report_age_method = "band_only"
    elif input_quality_tier == "adequate":
        if tumor_conf < 0.85:
            report_tumor_status = "uncertain tumor-status context"
            report_tumor_mode = "context"
        if report_site_mode == "exact" and site_conf < 0.80:
            report_site = _archetype_context(top_archetype, organ_family_label)
            report_site_mode = "context"
        if disease_family_conf < 0.80:
            report_disease_family = "uncertain phenotype family"
            report_disease_family_mode = "context"
        if disease_family_conf < 0.85 or disease_conf < 0.65 or support_match_n < 5:
            report_disease_label = "not stable enough to report"
            report_disease_label_mode = "withheld"
        if report_age_conf < 0.90 and report_age_conf >= 0.50:
            report_age_method = "band_only"
        elif report_age_conf < 0.50:
            report_age_band = "not stable enough to report"

    report_canonical_disease_label = normalize_disease_label(report_disease_label, "ontology")
    report_canonical_disease_label_mode = "exact" if report_disease_label_mode == "exact" else "withheld"
    if report_disease_label in ["not stable enough to report", "suppressed in high-confidence-only mode"]:
        report_canonical_disease_label = report_disease_label
        report_canonical_disease_label_mode = "withheld"
    if report_canonical_disease_label_mode == "exact":
        canonical_family = disease_family(
            str(report_canonical_disease_label).replace("_", " "),
            site_label,
            tumor_label,
        )
        if canonical_family != report_disease_family:
            report_canonical_disease_label = "not family-consistent enough to report"
            report_canonical_disease_label_mode = "withheld"
    # Targeted cluster-specific abstention rules for the largest residual
    # failure modes seen in held-out evaluation.
    if (
        report_site == "blood"
        and report_disease_family == "thoracic_tumor"
        and report_canonical_disease_label == "NSCLC"
    ):
        report_disease_family = "uncertain phenotype family"
        report_disease_family_mode = "withheld"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
    if (
        tumor_label == "tumor"
        and report_site_mode == "context"
        and isinstance(report_site, str)
        and "tissue context" in report_site
        and report_disease_family in {"digestive_tumor", "thoracic_tumor"}
        and report_canonical_disease_label in {"Gastric cancer", "pancreatic adenocarcinoma", "NSCLC"}
    ):
        report_disease_family = "uncertain phenotype family"
        report_disease_family_mode = "withheld"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
    if (
        str(site_label) == "other"
        and report_disease_family == "other_solid_tumor"
        and report_canonical_disease_label
        in {"Endometrioid adenocarcinoma, NOS", "Serous cystadenocarcinoma, NOS", "High-grade serous carcinoma"}
    ):
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
    # Breast-specific rescue: the remaining large false-positive cluster is a
    # subset of female lung adenocarcinoma samples that get pulled into a
    # breast-tumor neighborhood. True breast cases are typically much more
    # confident on both site and disease evidence, so we withhold only the
    # lower-confidence tail of the primary-mammary branch.
    if (
        report_canonical_disease_label == "breast_carcinoma"
        and str(report_disease_label) == "primary mammary tumor"
        and (site_conf < 0.85 or disease_conf < 0.85)
    ):
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
    # Structured evidence gate: only report a canonical disease entity when the
    # top-k neighborhood contains repeated support for that same canonical
    # entity. Cross-study diversity was too sparse to be useful, but raw
    # repeated local support is still predictive.
    if report_canonical_disease_label_mode == "exact" and support_match_n < 4:
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
    if (
        topk_gse_n <= 1
        and tumor_conf < 0.65
        and report_site_mode == "exact"
        and report_disease_family_mode == "exact"
        and report_canonical_disease_label_mode == "exact"
        and str(rescue["rescue_label"]) == "none"
    ):
        report_tumor_status = "uncertain epithelial or cohort-specific context"
        report_tumor_mode = "context"
        if organ_family_label == "digestive":
            report_site = f"{site_label} tissue context"
            report_site_mode = "context"
        report_disease_family = "uncertain phenotype family"
        report_disease_family_mode = "withheld"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
    if (
        top_archetype == "immune_dominant"
        and top_archetype_score > max(1.15 * second_archetype_score, 0.5)
        and report_site == "blood"
        and report_canonical_disease_label_mode == "withheld"
        and report_disease_family_mode != "exact"
    ):
        report_site = "immune-dominant tissue context"
        report_site_mode = "context"
        report_disease_family = "immune-dominant context"
        report_disease_family_mode = "context"
    if (
        tumor_label == "tumor"
        and tumor_conf < 0.60
        and thoracic_origin_prob < 0.75
        and str(rescue["rescue_label"]) == "none"
        and report_canonical_disease_label_mode == "withheld"
    ):
        report_tumor_status = "uncertain or activated non-malignant context"
        report_tumor_mode = "context"
    immune_heavy_prob = float(context_probs.get("immune_heavy_tumor_context", 0.0))
    activated_non_malignant_prob = float(context_probs.get("activated_non_malignant_immune_context", 0.0))
    fusion_label = str(fusion["fusion_label"])
    fusion_conf = float(fusion["fusion_conf"])
    if fusion_label == "activated_non_malignant_immune_context" and fusion_conf >= 0.22:
        if report_site in {"blood", "immune-dominant tissue context"} or top_archetype in {"immune_dominant", "hematologic_like"}:
            report_site = "blood" if report_site == "blood" else "immune-dominant tissue context"
            report_site_mode = "context"
        report_tumor_status = "uncertain or activated non-malignant context"
        report_tumor_mode = "context"
    rescue_label = str(rescue["rescue_label"])
    rescue_conf = float(rescue["rescue_conf"])
    if rescue_label == "breast_rescue" and rescue_conf >= 0.45:
        report_site = "breast tissue context"
        report_site_mode = "context"
        report_tumor_status = "tumor-like epithelial context" if tumor_label == "tumor" else "uncertain epithelial context"
        report_tumor_mode = "context"
        report_disease_family = "breast_tumor context"
        report_disease_family_mode = "context"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
    elif rescue_label == "hematologic_rescue" and rescue_conf >= 0.28:
        report_site = "hematologic tissue context"
        report_site_mode = "context"
        report_tumor_status = "tumor-like hematologic context" if tumor_label == "tumor" else "uncertain hematologic context"
        report_tumor_mode = "context"
        report_disease_family = "hematologic_malignancy context" if tumor_label == "tumor" else "hematologic context"
        report_disease_family_mode = "context"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
    elif rescue_label == "liver_rescue" and rescue_conf >= 0.35:
        report_site = "liver"
        report_site_mode = "exact"
        if tumor_label != "tumor":
            report_disease_family = "healthy_control"
            report_disease_family_mode = "context"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
    elif rescue_label == "brain_rescue" and rescue_conf >= 0.35:
        report_site = "brain"
        report_site_mode = "exact"
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
        if report_disease_family_mode != "exact":
            report_disease_family = "healthy_control"
            report_disease_family_mode = "context"
        if report_canonical_disease_label_mode != "exact":
            report_canonical_disease_label = "not stable enough to report"
            report_canonical_disease_label_mode = "withheld"
    if (
        topk_gse_n <= 1
        and report_canonical_disease_label_mode == "exact"
        and site_conf < 0.80
        and tumor_conf < 0.80
        and rescue_label == "none"
        and thoracic_origin_prob < 0.75
    ):
        report_canonical_disease_label = "not stable enough to report"
        report_canonical_disease_label_mode = "withheld"
        report_disease_label = "not stable enough to report"
        report_disease_label_mode = "withheld"
        if report_disease_family_mode == "exact":
            report_disease_family_mode = "context"
            report_disease_family = f"{report_disease_family} context" if "context" not in str(report_disease_family) else report_disease_family
        if report_site_mode == "exact":
            report_site_mode = "context"
            if report_site not in {"blood", "brain", "liver", "lung", "breast"}:
                report_site = f"{organ_family_label} tissue context"
        if report_tumor_mode == "exact":
            report_tumor_status = "tumor-like context" if tumor_label == "tumor" else "uncertain non-malignant context"
            report_tumor_mode = "context"
    if (
        tumor_label != "tumor"
        and report_site == "brain"
        and report_disease_family == "cns_tumor"
        and report_canonical_disease_label_mode == "withheld"
    ):
        report_disease_family = "healthy_control"
        report_disease_family_mode = "context"
    elif (
        fusion_label in {"immune_heavy_tumor_context", "thoracic_infiltrated_tumor_context"}
        and fusion_conf >= 0.35
        and tumor_label == "tumor"
        and (top_archetype in {"immune_dominant", "hematologic_like"} or report_site_mode != "exact")
    ):
        report_tumor_status = "immune-infiltrated tumor-like context"
        report_tumor_mode = "context"
        if report_site_mode != "exact":
            report_site = "immune-dominant tissue context"
            report_site_mode = "context"
        if report_disease_family_mode != "exact":
            report_disease_family = "immune-dominant context"
            report_disease_family_mode = "context"
    if (
        report_tumor_status == "immune-infiltrated tumor-like context"
        and top_archetype in {"immune_dominant", "hematologic_like"}
        and report_site in {"immune-dominant tissue context", "hematologic tissue context", "blood"}
        and report_canonical_disease_label_mode == "withheld"
        and report_disease_family_mode != "exact"
        and thoracic_origin_prob < 0.35
        and (
            immune_tumor_separator_prob < 0.45
            or tumor_conf < 0.75
        )
    ):
        report_tumor_status = "uncertain or activated non-malignant context"
        report_tumor_mode = "context"
        report_site = "hematologic tissue context" if top_archetype == "hematologic_like" else "immune-dominant tissue context"
        report_site_mode = "context"
        report_disease_family = "hematologic context" if top_archetype == "hematologic_like" else "immune-dominant context"
        report_disease_family_mode = "context"
    if (
        tumor_label == "tumor"
        and tumor_conf < 0.80
        and (
            report_disease_family in {"healthy_control", "immune-dominant context"}
            or report_site in {"immune-dominant tissue context", "blood"}
        )
    ):
        report_tumor_status = "uncertain or activated non-malignant context"
        report_tumor_mode = "context"
    origin_hypothesis, origin_hypothesis_kind = _infer_origin_hypothesis(
        top_archetype=top_archetype,
        archetype_scores=archetype_scores,
        tumor_conf=tumor_conf,
        thoracic_origin_prob=thoracic_origin_prob,
        report_site=report_site,
        report_disease_family=report_disease_family,
        report_canonical_disease_label_mode=report_canonical_disease_label_mode,
    )
    if (
        report_tumor_mode == "context"
        and origin_hypothesis != "none"
        and tumor_label == "tumor"
    ):
        report_tumor_status = "immune-infiltrated tumor-like context"
        report_tumor_mode = "context"
    elif (
        report_tumor_mode == "context"
        and origin_hypothesis == "none"
        and thoracic_origin_prob < 0.35
        and context_conf < 0.45
        and rescue_label == "none"
    ):
        report_tumor_status = "uncertain or activated non-malignant context"
        report_tumor_mode = "context"
        if report_site_mode != "exact":
            report_site = "immune-dominant tissue context"
            report_site_mode = "context"
        if report_disease_family_mode != "exact":
            report_disease_family = "immune-dominant context"
            report_disease_family_mode = "context"
    if report_canonical_disease_label_mode == "exact":
        disease_phrase = f"with {report_disease_family.replace('_', ' ')} features, most consistent with {report_canonical_disease_label}"
    else:
        disease_phrase = f"with {report_disease_family.replace('_', ' ')} features"
    if uncertainty["weak_fields"]:
        tail = f" Low-confidence fields: {', '.join(uncertainty['weak_fields'])}."
    else:
        tail = ""
    if report_age_method == "band_only" and report_age_conf >= 0.50:
        age_phrase = f"transcriptomic age-like state falls in the {report_age_band} range"
    elif report_age_conf >= 0.8:
        age_phrase = f"transcriptomic age-like state is around {int(round(report_age_years))} years"
    elif report_age_conf >= 0.5:
        age_phrase = f"transcriptomic age-like state falls in the {report_age_band} range"
    else:
        age_phrase = "age-like signal remains uncertain"
    if report_tumor_mode == "exact":
        sample_prefix = f"a {report_tumor_status} sample"
    elif report_tumor_mode == "context":
        sample_prefix = "a sample with uncertain tumor-state context"
    else:
        sample_prefix = "a sample"
    if input_quality_tier == "sparse":
        summary_text = (
            f"{lead} a sparse RNA profile with {report_site}. "
            f"Predicted sex is {sex_label}; {age_phrase}. "
            "Disease-level interpretation is withheld because too few matched genes support a stable phenotype call."
            f"{tail}"
        )
    elif input_quality_tier == "adequate":
        summary_text = (
            f"{lead} a partial RNA profile from {report_site} ({organ_family_label}) "
            f"{disease_phrase}. Predicted sex is {sex_label}; {age_phrase}. "
            "Fine disease output is conservative because the upload is only moderately complete."
            f"{tail}"
        )
    else:
        summary_text = (
            f"{lead} {sample_prefix} from {report_site} ({organ_family_label}) "
            f"{disease_phrase}. "
            f"Predicted sex is {sex_label}; {age_phrase}.{tail}"
        )
    if origin_hypothesis != "none":
        summary_text += f" Origin hypothesis: {origin_hypothesis}."
    if thoracic_origin_prob >= 0.75 and report_site_mode == "context":
        summary_text += " Dedicated thoracic-origin evidence is strong despite the immune-heavy neighborhood."
    if (
        fusion_label == "thoracic_infiltrated_tumor_context"
        and fusion_conf >= 0.55
        and (top_archetype in {"immune_dominant", "hematologic_like"} or report_site_mode != "exact")
    ):
        summary_text += " Fusion evidence supports an immune-infiltrated thoracic-like tumor context."
    elif fusion_label == "immune_heavy_tumor_context" and fusion_conf >= 0.35:
        summary_text += " Fusion evidence supports an immune-heavy tumor context rather than a purely activated non-malignant profile."
    if rescue_label != "none" and rescue_conf > 0:
        summary_text += f" Relative-expression rescue supports {rescue_label.replace('_', ' ')}."
    report_card = _build_report_card(
        uncertainty=uncertainty,
        report_age_years=report_age_years,
        report_age_method=report_age_method,
        report_age_conf=float(report_age_conf),
        sex_label=sex_label,
        sex_conf=float(max(sex_prob)),
        tumor_label=tumor_label,
        tumor_conf=tumor_conf,
        report_tumor_status=report_tumor_status,
        report_tumor_mode=report_tumor_mode,
        report_site=report_site,
        site_conf=float(site_conf),
        report_disease_family=report_disease_family,
        disease_family_conf=float(disease_family_conf),
        report_disease_label=report_disease_label,
        report_canonical_disease_label=report_canonical_disease_label,
        disease_conf=float(disease_conf),
        thoracic_origin_prob=float(thoracic_origin_prob),
        context_label=context_label,
        context_conf=float(context_conf),
        high_confidence_only=high_confidence_only,
    )
    support_profile = _compute_support_profile(
        input_quality_tier=input_quality_tier,
        report_tumor_status=report_tumor_status,
        report_tumor_mode=report_tumor_mode,
        report_site=report_site,
        report_site_mode=report_site_mode,
        report_disease_family=report_disease_family,
        report_disease_family_mode=report_disease_family_mode,
        report_canonical_disease_label=report_canonical_disease_label,
        report_canonical_disease_label_mode=report_canonical_disease_label_mode,
        pred_context_origin_fusion=fusion,
        origin_hypothesis=origin_hypothesis,
        pred_topk_gse_n=topk_gse_n,
        pred_site_conf=float(site_conf),
        pred_disease_family_conf=float(disease_family_conf),
        pred_disease_conf=float(disease_conf),
        pred_tumor_prob_head=float(tumor_prob_head),
    )
    if support_profile["support_level"] == "unsupported":
        report_card["withheld_calls"]["profile_support"] = {
            "reason": "; ".join(str(x) for x in support_profile["reasons"]),
        }
    elif support_profile["support_level"] == "mixed_interpretable":
        report_card["context_calls"]["profile_support"] = {
            "value": "mixed but interpretable",
            "confidence": float(support_profile["support_confidence"]),
            "reason": "; ".join(str(x) for x in support_profile["reasons"]),
            "fallback": "coarse phenotype context only",
        }
    else:
        report_card["safe_calls"]["profile_support"] = {
            "value": "supported phenotype profile",
            "confidence": float(support_profile["support_confidence"]),
            "reason": "; ".join(str(x) for x in support_profile["reasons"]),
        }

    input_payload = {
        "n_rows_parsed": int(len(raw)),
        "n_unique_genes": int(raw.index.nunique()),
        "n_selected_genes_matched": int(len(overlap)),
        "selected_gene_coverage": coverage,
        "input_quality_tier": input_quality_tier,
        "top_archetype": top_archetype,
        "top_archetype_score": top_archetype_score,
        "value_mode": resolved_mode,
        "input_sufficient": True,
    }
    predictions_payload = {
            "pred_age_years_adapter": age_years,
            "pred_age_years_knn": age_knn,
            "pred_age_conf_knn": age_knn_conf,
            "report_age_years": report_age_years,
            "report_age_method": report_age_method,
            "report_age_conf": report_age_conf,
            "report_age_band": report_age_band,
            "report_age_mode": (
                "exact"
                if report_age_conf >= 0.80 and report_age_method != "band_only"
                else ("band" if report_age_conf >= 0.50 else "withheld")
            ),
            "pred_sex_head": sex_label,
            "pred_sex_head_conf": float(max(sex_prob)),
            "pred_sex_knn": sex_knn_label,
            "pred_sex_knn_conf": sex_knn_conf,
            "pred_tumor_status_head": tumor_label,
            "pred_tumor_prob_head": float(tumor_prob_head),
            "report_tumor_status": report_tumor_status,
            "report_tumor_mode": report_tumor_mode,
            "pred_anatomical_site": site_label,
            "pred_site_conf": site_conf,
            "pred_disease_label": disease_label,
            "pred_canonical_disease_label": pred_canonical_disease_label,
            "pred_disease_conf": disease_conf,
            "pred_disease_family": disease_family_label,
            "pred_disease_family_conf": disease_family_conf,
            "pred_canonical_support_match_n": support_match_n,
            "pred_canonical_support_gse_n": support_gse_n,
            "pred_topk_gse_n": topk_gse_n,
            "pred_expression_archetype": top_archetype,
            "pred_expression_archetype_score": top_archetype_score,
            "pred_relative_signature_scores": relative_signature_scores,
            "pred_thoracic_origin_prob": thoracic_origin_prob,
            "pred_context_label": context_label,
            "pred_context_conf": context_conf,
            "pred_context_probs": context_probs,
            "pred_immune_heavy_tumor_context_prob": immune_heavy_prob,
            "pred_activated_non_malignant_immune_context_prob": activated_non_malignant_prob,
            "pred_immune_tumor_separator_prob": immune_tumor_separator_prob,
            "pred_context_origin_fusion": fusion,
            "pred_relative_signature_rescue": rescue,
            "origin_hypothesis": origin_hypothesis,
            "origin_hypothesis_kind": origin_hypothesis_kind,
            "pred_support_level": support_profile["support_level"],
            "pred_support_confidence": support_profile["support_confidence"],
            "pred_support_reasons": support_profile["reasons"],
            "pred_organ_family": organ_family_label,
            "report_site": report_site,
            "report_site_mode": report_site_mode,
            "report_disease_family": report_disease_family,
            "report_disease_family_mode": report_disease_family_mode,
            "report_disease_label": report_disease_label,
            "report_disease_label_mode": report_disease_label_mode,
            "report_canonical_disease_label": report_canonical_disease_label,
            "report_canonical_disease_label_mode": report_canonical_disease_label_mode,
        }
    explainer = _build_openworld_explainer(
        input_payload=input_payload,
        predictions=predictions_payload,
        uncertainty=uncertainty,
        archetype_scores=archetype_scores,
        relative_signature_scores=relative_signature_scores,
        support=support,
        support_profile=support_profile,
        summary_text=summary_text,
    )
    return {
        "input": input_payload,
        "predictions": predictions_payload,
        "distributions": {
            "site": site_dist,
            "disease_label": disease_dist,
            "disease_family": disease_family_dist,
            "sex_knn": sex_knn_dist,
        },
        "nearest_neighbors": support,
        "uncertainty": uncertainty,
        "expression_archetypes": archetype_scores,
        "report_card": report_card,
        "summary_text": summary_text,
        "explainer": explainer,
        "caveat": "This is phenotype-similarity inference from the learned bulk RNA representation, not a calibrated prospective disease-risk model.",
    }


def run_unknown_sample_openworld(
    counts_text: str,
    top_k: int = DEFAULT_TOP_K,
    value_mode: str = "auto",
    high_confidence_only: bool = False,
) -> dict:
    primary = run_unknown_sample(
        counts_text,
        top_k=top_k,
        value_mode=value_mode,
        high_confidence_only=high_confidence_only,
    )
    if not primary["input"].get("input_sufficient", False):
        primary["openworld"] = {
            "mode": "single",
            "chosen_route": "primary",
            "reason": "input insufficient; skipping alternate route",
        }
        return primary

    raw = _parse_gene_count_lines(counts_text)
    logcpm_text = _to_logcpm_text(raw)
    if not logcpm_text:
        primary["openworld"] = {
            "mode": "single",
            "chosen_route": "primary",
            "reason": "could not construct logCPM alternate route",
        }
        return primary

    alt = run_unknown_sample(
        logcpm_text,
        top_k=top_k,
        value_mode="log1p",
        high_confidence_only=high_confidence_only,
    )

    def score(payload: dict) -> float:
        pred = payload.get("predictions", {})
        return float(
            pred.get("pred_site_conf", 0.0)
            + pred.get("pred_disease_family_conf", 0.0)
            + pred.get("pred_disease_conf", 0.0)
            + 0.05 * float(pred.get("pred_canonical_support_match_n", 0))
        )

    def reported(payload: dict, key: str) -> str:
        return str(payload.get("predictions", {}).get(key, "withheld"))

    primary_score = score(primary)
    alt_score = score(alt)
    top_archetype = str(primary.get("input", {}).get("top_archetype", "unknown"))
    primary_org = str(primary.get("predictions", {}).get("pred_organ_family", "unknown"))
    alt_org = str(alt.get("predictions", {}).get("pred_organ_family", "unknown"))
    primary_site = reported(primary, "report_site")
    alt_site = reported(alt, "report_site")
    primary_family = reported(primary, "report_disease_family")
    alt_family = reported(alt, "report_disease_family")
    primary_can = reported(primary, "report_canonical_disease_label")
    alt_can = reported(alt, "report_canonical_disease_label")

    chosen = primary
    chosen_route = "primary"
    reason = "primary route retained"

    if primary_can == alt_can and "not stable enough" not in primary_can:
        chosen = primary if primary_score >= alt_score else alt
        chosen_route = "consensus"
        reason = "both routes agree on canonical disease label"
    elif (
        top_archetype == "thoracic_like"
        and alt_org == "thoracic"
        and primary_org != "thoracic"
        and alt_score >= primary_score - 0.05
    ):
        chosen = alt
        chosen_route = "logcpm-archetype"
        reason = "thoracic-like expression archetype favors the logCPM thoracic route"
    elif primary_org == alt_org and primary_family == alt_family and alt_score > primary_score + 0.10:
        chosen = alt
        chosen_route = "logcpm"
        reason = "same organ/family but logCPM route has stronger evidence"
    elif primary_site != alt_site and primary_org != alt_org:
        chosen = primary if primary_score >= alt_score else alt
        chosen_route = "degraded-consensus"
        reason = "routes disagree on site/organ; degrading disease outputs"
        chosen["predictions"]["report_disease_family"] = "uncertain phenotype family"
        chosen["predictions"]["report_disease_family_mode"] = "withheld"
        chosen["predictions"]["report_disease_label"] = "not stable enough to report"
        chosen["predictions"]["report_disease_label_mode"] = "withheld"
        chosen["predictions"]["report_canonical_disease_label"] = "not stable enough to report"
        chosen["predictions"]["report_canonical_disease_label_mode"] = "withheld"
        chosen["summary_text"] = (
            f"Moderate evidence suggests a sample from {chosen['predictions']['report_site']} "
            f"({chosen['predictions']['pred_organ_family']}), but alternate input normalizations disagree on phenotype context. "
            "Disease-level interpretation is withheld pending stronger evidence."
        )
        if "report_card" in chosen:
            chosen["report_card"]["withheld_calls"]["canonical_disease_label"] = {
                "reason": "routes disagree under alternative normalization"
            }
            chosen["report_card"]["withheld_calls"]["disease_label"] = {
                "reason": "routes disagree under alternative normalization"
            }
            chosen["report_card"]["context_calls"]["disease_family_context"] = {
                "value": "uncertain phenotype family",
                "confidence": 0.0,
                "reason": "routes disagree under alternative normalization",
                "fallback": "uncertain phenotype family",
            }
        primary_pred = primary.get("predictions", {})
        alt_pred = alt.get("predictions", {})
        candidate_site_strings = {
            str(primary_pred.get("report_site", "")),
            str(alt_pred.get("report_site", "")),
        }
        candidate_family_strings = {
            str(primary_pred.get("report_disease_family", "")),
            str(alt_pred.get("report_disease_family", "")),
        }
        candidate_rescue_labels = {
            str((primary_pred.get("pred_relative_signature_rescue") or {}).get("rescue_label", "none")),
            str((alt_pred.get("pred_relative_signature_rescue") or {}).get("rescue_label", "none")),
        }
        candidate_origin_hypotheses = {
            str(primary_pred.get("origin_hypothesis", "none")),
            str(alt_pred.get("origin_hypothesis", "none")),
        }
        if any(s in {"blood", "immune-dominant tissue context", "hematologic tissue context"} for s in candidate_site_strings):
            chosen["predictions"]["report_site"] = "immune-dominant tissue context"
            chosen["predictions"]["report_site_mode"] = "context"
            if "report_card" in chosen:
                chosen["report_card"]["context_calls"]["site_context"] = {
                    "value": "immune-dominant tissue context",
                    "confidence": 0.0,
                    "reason": "routes disagree; one route points to hematologic or immune-dominant context",
                    "fallback": "immune-dominant tissue context",
                }
        if any("hematologic" in fam for fam in candidate_family_strings) or "hematologic_rescue" in candidate_rescue_labels:
            chosen["predictions"]["report_disease_family"] = "hematologic_malignancy context"
            chosen["predictions"]["report_disease_family_mode"] = "context"
            if "report_card" in chosen:
                chosen["report_card"]["context_calls"]["disease_family_context"] = {
                    "value": "hematologic_malignancy context",
                    "confidence": 0.0,
                    "reason": "routes disagree, but one route supports hematologic context",
                    "fallback": "hematologic_malignancy context",
                }
        if (
            all(label == "none" for label in candidate_rescue_labels)
            and all(h == "none" for h in candidate_origin_hypotheses)
            and "hematologic" not in " ".join(candidate_family_strings)
        ):
            chosen["predictions"]["report_tumor_status"] = "unsupported mixed biological context"
            chosen["predictions"]["report_tumor_mode"] = "context"
            chosen["predictions"]["report_site"] = "mixed tissue-of-origin context"
            chosen["predictions"]["report_site_mode"] = "context"
            chosen["predictions"]["report_disease_family"] = "not stable enough to report"
            chosen["predictions"]["report_disease_family_mode"] = "withheld"
            chosen["summary_text"] = (
                "Weak evidence suggests this RNA profile is unsupported by the current phenotype atlas: "
                "alternate input normalizations disagree on tissue-of-origin and no stable rescue signal is present. "
                "Only coarse uncertainty-aware outputs are retained."
            )
            if "report_card" in chosen:
                chosen["report_card"]["withheld_calls"]["profile_support"] = {
                    "reason": "mixed biological context / unstable route agreement"
                }
    elif alt_score > primary_score + 0.20:
        chosen = alt
        chosen_route = "logcpm"
        reason = "logCPM route has materially stronger evidence"

    pred = chosen.get("predictions", {})
    if (
        top_archetype in {"immune_dominant", "hematologic_like"}
        and str(pred.get("report_site")) == "blood"
        and str(pred.get("report_canonical_disease_label")).startswith("not stable enough")
        and str(pred.get("report_disease_family_mode")) != "exact"
    ):
        rescue_label = str((pred.get("pred_relative_signature_rescue") or {}).get("rescue_label", "none"))
        if rescue_label == "hematologic_rescue":
            pred["report_site"] = "hematologic tissue context"
            pred["report_site_mode"] = "context"
            pred["report_disease_family"] = "hematologic_malignancy context"
            pred["report_disease_family_mode"] = "context"
            profile_label = "hematologic-like"
        else:
            pred["report_site"] = "immune-dominant tissue context"
            pred["report_site_mode"] = "context"
            pred["report_disease_family"] = "immune-dominant context"
            pred["report_disease_family_mode"] = "context"
            profile_label = "immune-dominant"
        origin_hypothesis = str(pred.get("origin_hypothesis", "none"))
        origin_tail = ""
        if origin_hypothesis and origin_hypothesis != "none":
            origin_tail = f" Origin hypothesis: {origin_hypothesis}."
        chosen["summary_text"] = (
            f"Moderate evidence suggests a {profile_label} RNA profile with uncertain tissue-of-origin. "
            f"Predicted sex is {pred.get('pred_sex_head', 'unknown')}; disease-level interpretation is withheld pending stronger tissue context."
            f"{origin_tail}"
        )
        thoracic_prob = float(pred.get("pred_thoracic_origin_prob", 0.0) or 0.0)
        if thoracic_prob >= 0.75:
            chosen["summary_text"] += " Dedicated thoracic-origin support remains strong despite the immune-heavy neighborhood."
        if "report_card" in chosen:
            chosen["report_card"]["context_calls"]["site_context"] = {
                "value": pred["report_site"],
                "confidence": 0.0,
                "reason": "expression archetype indicates hematologic or immune-dominant context and disease-level routes remain unstable",
                "fallback": pred["report_site"],
            }
            chosen["report_card"]["context_calls"]["disease_family_context"] = {
                "value": pred["report_disease_family"],
                "confidence": 0.0,
                "reason": "expression archetype indicates hematologic or immune-dominant context and disease-level routes remain unstable",
                "fallback": pred["report_disease_family"],
            }

    chosen["openworld"] = {
        "mode": "dual-path",
        "chosen_route": chosen_route,
        "reason": reason,
        "primary_route": {
            "value_mode": primary["input"].get("value_mode"),
            "report_site": primary_site,
            "report_disease_family": primary_family,
            "report_canonical_disease_label": primary_can,
            "score": primary_score,
        },
        "alternate_route": {
            "value_mode": alt["input"].get("value_mode"),
            "report_site": alt_site,
            "report_disease_family": alt_family,
            "report_canonical_disease_label": alt_can,
            "score": alt_score,
        },
    }
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-file", required=True, help="Text/CSV/TSV file with gene,count rows.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--value-mode", choices=["auto", "raw_count", "log1p"], default="auto")
    args = parser.parse_args()

    text = Path(args.counts_file).read_text(encoding="utf-8")
    payload = run_unknown_sample_openworld(text, top_k=args.top_k, value_mode=args.value_mode)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
