#!/usr/bin/env python3
"""Semantic-first unknown bulk RNA explainer using fused MHA semantic embedding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from family_taxonomy import disease_family, organ_family
from repro_paths import BME_CODE_DIR as MODULE_DIR
from repro_paths import OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[2]
BEST_CFG = OUTPUT_ROOT / "semantic_mainline_best_20260417.json"
EVIDENCE_PRIORS_PATH = OUTPUT_ROOT / "semantic_state_evidence_priors_20260418.json"
EVIDENCE_SCORER_CANDIDATES = [
    OUTPUT_ROOT / "semantic_state_evidence_scorer_max" / "model.pkl",
    OUTPUT_ROOT / "train_semantic_state_evidence_scorer_max_20260418" / "model.pkl",
    OUTPUT_ROOT / "train_semantic_state_evidence_scorer_broad_rerun_20260418" / "model.pkl",
    OUTPUT_ROOT / "train_semantic_state_evidence_scorer_broad_20260418" / "model.pkl",
    OUTPUT_ROOT / "train_semantic_state_evidence_scorer_20260418" / "model.pkl",
]
TRAIN_SCRIPT = MODULE_DIR / "train_semantic_prototype_attention.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


def _age_band(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "unknown"
    v = float(v)
    if v < 45:
        return "younger-adult"
    if v < 70:
        return "midlife-adult"
    return "older-adult"


def _contains_any(text: str, patterns: list[str]) -> bool:
    s = str(text).lower()
    return any(p in s for p in patterns)


def _context_tags_from_row(row: pd.Series) -> list[str]:
    disease = str(row.get("feat_disease_label", ""))
    site = str(row.get("feat_anatomical_site", ""))
    tumor = str(row.get("feat_tumor_status", ""))
    semantic = str(row.get("metadata_text", ""))
    blob = " | ".join([disease, site, tumor, semantic]).lower()
    tags: list[str] = []
    if tumor in {"tumor", "metastatic"} or "tumor" in blob or "carcinoma" in blob or "adenocarcinoma" in blob:
        tags.append("tumor-associated state")
    if _contains_any(blob, ["covid", "viral", "virus", "infect", "sepsis", "influenza", "hiv", "hepatitis"]):
        tags.append("infectious or viral stress")
    if _contains_any(blob, ["inflamm", "autoimmune", "crohn", "colitis", "lupus", "arthritis", "allergy", "asthma", "ibd"]):
        tags.append("inflammatory or autoimmune stress")
    if _contains_any(blob, ["obesity", "obese", "diabetes", "metabolic", "nafld", "nash", "steatosis", "fatty liver"]):
        tags.append("metabolic stress")
    if _contains_any(blob, ["blood", "pbmc", "leukocyte", "hematologic", "lymph", "marrow", "immune"]):
        tags.append("immune or hematologic dominance")
    return tags


def _weighted_knn_regression(values: np.ndarray, scores: np.ndarray) -> tuple[float | None, float]:
    vals = np.asarray(values, dtype=np.float32)
    mask = ~np.isnan(vals)
    if not mask.any():
        return None, 0.0
    w = np.clip(np.asarray(scores, dtype=np.float32)[mask], 0, None) + 1e-6
    return float(np.average(vals[mask], weights=w)), float(np.mean(scores[mask]))


@lru_cache(maxsize=1)
def _load_state_evidence_priors() -> dict[str, dict[str, dict[str, float]]]:
    if os.getenv("SEMANTIC_DISABLE_EVIDENCE_PRIORS", "0") == "1":
        return {"family_priors": {}, "subprofile_priors": {}}
    if not EVIDENCE_PRIORS_PATH.exists():
        return {"family_priors": {}, "subprofile_priors": {}}
    try:
        payload = json.loads(EVIDENCE_PRIORS_PATH.read_text())
    except Exception:
        return {"family_priors": {}, "subprofile_priors": {}}
    return {
        "family_priors": payload.get("family_priors", {}) or {},
        "subprofile_priors": payload.get("subprofile_priors", {}) or {},
    }


@lru_cache(maxsize=1)
def _load_state_evidence_scorer():
    if os.getenv("SEMANTIC_DISABLE_EVIDENCE_SCORER", "0") == "1":
        return None
    for path in EVIDENCE_SCORER_CANDIDATES:
        if not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            continue
    return None


def _state_evidence_model_features(
    state_family: str,
    state_subprofile: str,
    semantic_route: str,
    explanation_mode: str,
    anchor_adjusted: bool,
    evidence_highlights: list[dict[str, object]],
) -> dict[str, object]:
    feat: dict[str, object] = {
        "state_family": state_family,
        "state_subprofile": state_subprofile,
        "route": semantic_route,
        "mode": explanation_mode,
        "anchor_adjusted": int(bool(anchor_adjusted)),
    }
    for kind in [
        "semantic_majority",
        "anchor_context",
        "neighbor_context",
        "age_driver_overlap",
        "tension",
        "anchor_override",
    ]:
        item = next((x for x in evidence_highlights if str(x.get("kind")) == kind), {})
        feat[f"strength_{kind}"] = float(item.get("strength", 0.0) or 0.0)
    return feat


def _counts_to_text(counts: dict[str, float]) -> str:
    return "\n".join(f"{g},{v}" for g, v in sorted(counts.items()))


@lru_cache(maxsize=1)
def _load_structured_anchor_runner():
    from infer_unknown_rna_sample import run_unknown_sample_openworld

    return run_unknown_sample_openworld


def _compute_structured_anchor(counts: dict[str, float]) -> dict:
    runner = _load_structured_anchor_runner()
    payload = runner(_counts_to_text(counts), top_k=10, value_mode="auto", high_confidence_only=False)
    pred = payload.get("predictions", {})
    expl = payload.get("explainer", {})
    support = expl.get("support", {})
    context = expl.get("context", {})
    phenotype = expl.get("phenotype", {})
    return {
        "support_level": str(support.get("level", pred.get("pred_support_level", "unknown"))),
        "support_confidence": float(support.get("confidence", pred.get("pred_support_confidence", 0.0)) or 0.0),
        "context_state": str(context.get("state", pred.get("pred_context_label", "unknown"))),
        "tissue_context": str(context.get("tissue_context", "unknown")),
        "tumor_status": str((phenotype.get("tumor_status") or {}).get("value", pred.get("report_tumor_status", "unknown"))),
        "tumor_mode": str((phenotype.get("tumor_status") or {}).get("mode", pred.get("report_tumor_mode", "unknown"))),
        "site": str((phenotype.get("site") or {}).get("value", pred.get("report_site", "unknown"))),
        "disease_family": str((phenotype.get("disease_family") or {}).get("value", pred.get("report_disease_family", "unknown"))),
    }


def _apply_anchor_demote(semantic_consensus: dict, anchor: dict) -> dict:
    route = str(semantic_consensus.get("route", "unknown"))
    if route != "stable_semantic_frame":
        semantic_consensus["anchor_adjusted"] = False
        return semantic_consensus

    anchor_context = str(anchor.get("context_state", "unknown"))
    anchor_tumor = str(anchor.get("tumor_status", "unknown")).lower()
    anchor_tumor_mode = str(anchor.get("tumor_mode", "unknown"))
    anchor_support = str(anchor.get("support_level", "unknown"))

    non_malignant_anchor = (
        "non-malignant" in anchor_tumor
        or anchor_context == "activated_non_malignant_immune_context"
        or (anchor_tumor_mode != "exact" and anchor_support in {"mixed_interpretable", "unsupported"})
    )
    if non_malignant_anchor and semantic_consensus.get("tumor_status") == "tumor":
        semantic_consensus = dict(semantic_consensus)
        semantic_consensus["route"] = "mixed_semantic_frame"
        semantic_consensus["route_subtype"] = "anchor_context_override"
        semantic_consensus["anchor_adjusted"] = True
        semantic_consensus["reason"] = (
            "semantic tumor-like frame was downgraded because the structured anchor supports a non-malignant or mixed immune context"
        )
    else:
        semantic_consensus["anchor_adjusted"] = False
    return semantic_consensus


def _build_semantic_explanation(
    semantic_consensus: dict,
    structured_anchor: dict,
    top_prototype: dict,
    neighbors: list[dict],
    age_like_state: dict,
) -> dict:
    disable_state_map = os.getenv("SEMANTIC_DISABLE_STATE_MAP", "0") == "1"
    disable_disease_card = os.getenv("SEMANTIC_DISABLE_DISEASE_CARD", "0") == "1"
    route = str(semantic_consensus.get("route", "unknown"))
    proto_site_family = str(semantic_consensus.get("prototype_site_family", "unknown"))
    proto_disease_family = str(semantic_consensus.get("prototype_disease_family", "unknown"))
    voted_site_family = str(semantic_consensus.get("site_family", "unknown"))
    voted_disease_family = str(semantic_consensus.get("disease_family", "unknown"))
    anchor_context = str(structured_anchor.get("context_state", "unknown"))
    anchor_tissue = str(structured_anchor.get("tissue_context", "unknown"))
    anchor_support = str(structured_anchor.get("support_level", "unknown"))
    anchor_tumor = str(structured_anchor.get("tumor_status", "unknown"))
    anchor_support_conf = float(structured_anchor.get("support_confidence", 0.0) or 0.0)
    age_band = str(age_like_state.get("band", "unknown"))
    age_drivers = list(age_like_state.get("possible_drivers", []) or [])
    semantic_strength = float(
        max(
            semantic_consensus.get("site_family_confidence", 0.0),
            semantic_consensus.get("tumor_confidence", 0.0),
            semantic_consensus.get("disease_family_confidence", 0.0),
        )
    )
    neighbor_sites = [organ_family(str(n.get("site_anchor", "unknown"))) for n in neighbors]
    neighbor_tags = [str(tag) for n in neighbors for tag in (n.get("context_tags", []) or [])]
    neighbor_diseases = [
        disease_family(
            str(n.get("disease_anchor", "unknown")),
            str(n.get("site_anchor", "unknown")),
            str(n.get("tumor_anchor", "unknown")),
        )
        for n in neighbors
    ]

    if route == "stable_semantic_frame":
        explanation_mode = "stable_consensus"
        primary = f"stable {voted_site_family} / {voted_disease_family} semantic frame"
        counter = "no major contradiction between prototype and semantic neighbors"
    elif route == "mixed_semantic_frame":
        subtype = str(semantic_consensus.get("route_subtype", "mixed_tumor_semantics"))
        explanation_mode = "conflict_explainer"
        if subtype == "prototype_neighbor_family_conflict" or proto_site_family != voted_site_family:
            explanation_mode = "family_conflict_explainer"
            primary = (
                f"prototype suggests {proto_site_family} / {proto_disease_family}, "
                f"but semantic neighbors vote toward {voted_site_family} / {voted_disease_family}"
            )
            counter = "prototype and neighbor tissue families disagree"
        elif subtype == "anchor_context_override" or semantic_consensus.get("anchor_adjusted", False):
            explanation_mode = "context_override_explainer"
            if "activated_non_malignant_immune_context" in anchor_context:
                primary = (
                    f"tumor-like semantic frame centered on {proto_site_family} / {proto_disease_family}, "
                    "but the broader transcriptomic context looks like activated non-malignant or context-heavy biology"
                )
            elif anchor_context == "clean_context":
                primary = (
                    f"tumor-like semantic frame centered on {proto_site_family} / {proto_disease_family}, "
                    "but the structured anchor looks cleaner and less malignant than the semantic tumor frame"
                )
            else:
                primary = f"tumor-like semantic frame centered on {proto_site_family} / {proto_disease_family}"
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "this looks more like context-dominant tumor mimicry or mixed biology than a stable malignant frame"
            )
        else:
            primary = f"tumor-like semantic frame centered on {voted_site_family} / {voted_disease_family}"
            counter = "broad tissue family is consistent, but disease-level semantics remain mixed"
    else:
        explanation_mode = "unsupported_semantics"
        primary = "semantic neighbors do not support a stable broad biological frame"
        counter = "prototype and neighbor semantics disagree too strongly"

    tag_votes: dict[str, int] = {}
    for tag in neighbor_tags:
        tag_votes[tag] = tag_votes.get(tag, 0) + 1
    dominant_neighbor_tags = [k for k, _ in sorted(tag_votes.items(), key=lambda kv: (-kv[1], kv[0]))[:4]]
    age_driver_overlap = [tag for tag in age_drivers if tag in dominant_neighbor_tags]
    total_tag_votes = max(sum(tag_votes.values()), 1)
    dominant_tag_strength = float(sum(tag_votes.get(tag, 0) for tag in dominant_neighbor_tags[:3]) / total_tag_votes) if dominant_neighbor_tags else 0.0
    age_overlap_strength = float(len(age_driver_overlap) / max(len(age_drivers), 1)) if age_drivers else 0.0
    resolved_disease_family = voted_disease_family
    disease_semantic_status = "stable" if route == "stable_semantic_frame" else ("mixed" if route == "mixed_semantic_frame" else "unsupported")
    disease_semantic_note = "broad disease-family semantics are usable"
    disease_semantic_subprofile = "stable_consensus"
    if disable_disease_card:
        if route == "unstable_semantic_frame":
            resolved_disease_family = voted_disease_family
            disease_semantic_status = "unsupported"
            disease_semantic_note = "raw disease-family semantics are unstable and remain unresolved in this ablation"
            disease_semantic_subprofile = "raw_disease_semantics_unsupported"
        elif route == "mixed_semantic_frame":
            resolved_disease_family = voted_disease_family
            disease_semantic_status = "mixed"
            disease_semantic_note = "raw disease-family semantics are left unresolved in this ablation"
            disease_semantic_subprofile = "raw_disease_semantics_mixed"
        else:
            resolved_disease_family = voted_disease_family
            disease_semantic_status = "stable"
            disease_semantic_note = "raw disease-family semantics are used directly in this ablation"
            disease_semantic_subprofile = "raw_disease_semantics_stable"
    elif route == "unstable_semantic_frame":
        resolved_disease_family = "unsupported_disease_semantics"
        disease_semantic_status = "unsupported"
        disease_semantic_note = "disease-level semantics are too unstable to interpret"
        disease_semantic_subprofile = "unsupported_disease_semantics"
    elif (
        route == "stable_semantic_frame"
        and voted_disease_family == "other_solid_tumor"
        and (
            (anchor_context == "clean_context" and anchor_tumor.lower() == "non_tumor")
            or anchor_tumor.lower() == "unsupported mixed biological context"
        )
    ):
        resolved_disease_family = "tumor_like_other_solid_overlap"
        disease_semantic_status = "mixed"
        disease_semantic_note = "solid-tumor-like disease semantics are present, but the structured anchor is too clean or non-malignant to treat them as a stable other_solid_tumor frame"
        if anchor_context == "clean_context":
            disease_semantic_subprofile = "other_solid_overlap_clean_anchor"
        else:
            disease_semantic_subprofile = "other_solid_overlap_generic"
    elif (
        route == "stable_semantic_frame"
        and voted_disease_family == "healthy_control"
        and (
            "immune-infiltrated" in anchor_tumor.lower()
            or anchor_tumor.lower() == "unsupported mixed biological context"
            or anchor_context == "unsupported"
        )
    ):
        resolved_disease_family = "healthy_control_like_overlap"
        disease_semantic_status = "mixed"
        disease_semantic_note = "healthy-control-like disease semantics are present, but the broader anchor is too activated or unsupported to treat them as a clean stable baseline state"
        if "immune-infiltrated" in anchor_tumor.lower():
            disease_semantic_subprofile = "healthy_control_overlap_immune_activation"
        else:
            disease_semantic_subprofile = "healthy_control_overlap_unsupported_context"
    elif (
        route == "stable_semantic_frame"
        and voted_disease_family == "hematologic_solid_tumor"
        and anchor_tumor.lower() == "unsupported mixed biological context"
    ):
        resolved_disease_family = "mixed_hematologic_solid_tumor"
        disease_semantic_status = "mixed"
        disease_semantic_note = "hematologic solid-tumor-like disease semantics are present, but the structured anchor is too unsupported to treat them as a stable hematologic solid-tumor frame"
        disease_semantic_subprofile = "hematologic_solid_overlap_unsupported_context"
    elif route == "mixed_semantic_frame":
        disease_semantic_status = "mixed"
        if voted_disease_family == "other_solid_tumor":
            resolved_disease_family = "tumor_like_other_solid_overlap"
            disease_semantic_note = "broad tumor-like semantics are present, but disease-family support is too mixed to treat as stable other_solid_tumor"
            if anchor_context == "clean_context":
                disease_semantic_subprofile = "other_solid_overlap_clean_anchor"
            elif anchor_context == "activated_non_malignant_immune_context":
                disease_semantic_subprofile = "other_solid_overlap_non_malignant_activation"
            elif "immune" in anchor_context or "infiltrated" in anchor_tumor.lower():
                disease_semantic_subprofile = "other_solid_overlap_immune_infiltrated"
            else:
                disease_semantic_subprofile = "other_solid_overlap_generic"
        elif voted_disease_family == "digestive_tumor":
            resolved_disease_family = "digestive_tumor"
            disease_semantic_note = "digestive tumor-like disease semantics are present, but the broader anchor and context remain too mixed to treat them as a stable digestive tumor frame"
            if anchor_context == "clean_context":
                disease_semantic_subprofile = "digestive_overlap_clean_anchor"
            elif anchor_context == "activated_non_malignant_immune_context" or "uncertain or activated non-malignant context" in anchor_tumor.lower():
                disease_semantic_subprofile = "digestive_overlap_non_malignant_activation"
            elif "immune" in anchor_context or "infiltrated" in anchor_tumor.lower():
                disease_semantic_subprofile = "digestive_overlap_immune_infiltrated"
            else:
                disease_semantic_subprofile = "digestive_overlap_generic"
        elif voted_disease_family in {"hematologic_malignancy", "hematologic_solid_tumor"} and "activated_non_malignant_immune_context" in anchor_context:
            resolved_disease_family = f"mixed_{voted_disease_family}"
            disease_semantic_note = "hematologic tumor-like disease semantics are present, but they remain mixed under non-malignant immune override"
            if "immune-infiltrated" in anchor_tumor.lower():
                disease_semantic_subprofile = "hematologic_malignancy_overlap_infiltrated"
            else:
                disease_semantic_subprofile = "hematologic_malignancy_overlap_non_malignant"
        elif voted_disease_family in {"breast_tumor", "breast_non_tumor", "thoracic_tumor", "digestive_solid_tumor", "cns_tumor"}:
            resolved_disease_family = voted_disease_family
            disease_semantic_note = f"{voted_disease_family} semantics are present, but the broader anchor and context remain too mixed to stabilize them as a clean disease-semantic call"
            prefix_map = {
                "breast_tumor": "breast",
                "breast_non_tumor": "breast_non_malignant",
                "thoracic_tumor": "thoracic",
                "digestive_solid_tumor": "digestive_solid",
                "cns_tumor": "cns",
            }
            prefix = prefix_map.get(voted_disease_family, voted_disease_family)
            if anchor_context == "clean_context":
                disease_semantic_subprofile = f"{prefix}_overlap_clean_anchor"
            elif anchor_context == "activated_non_malignant_immune_context" or "uncertain or activated non-malignant context" in anchor_tumor.lower():
                disease_semantic_subprofile = f"{prefix}_overlap_non_malignant_activation"
            elif "immune" in anchor_context or "infiltrated" in anchor_context or "infiltrated" in anchor_tumor.lower():
                disease_semantic_subprofile = f"{prefix}_overlap_immune_infiltrated"
            else:
                disease_semantic_subprofile = f"{prefix}_overlap_generic"
        else:
            if voted_disease_family == "healthy_control":
                if anchor_tumor.lower() == "unsupported mixed biological context":
                    disease_semantic_subprofile = "healthy_control_overlap_unsupported_context"
                    disease_semantic_note = "healthy-control-like disease semantics are present, but the broader context is too unsupported to treat them as a stable baseline state"
                elif (
                    "uncertain or activated non-malignant context" in anchor_tumor.lower()
                    or "immune-infiltrated" in anchor_tumor.lower()
                    or anchor_context in {"activated_non_malignant_immune_context", "immune_heavy_tumor_context"}
                ):
                    disease_semantic_subprofile = "healthy_control_overlap_immune_activation"
                    disease_semantic_note = "healthy-control-like disease semantics are present, but activation is too strong to treat them as a clean stable baseline state"
                else:
                    disease_semantic_subprofile = "healthy_control_overlap_generic"
                    disease_semantic_note = "healthy-control-like disease semantics are present, but disease support remains mixed"
            elif voted_site_family == "hematologic":
                if voted_disease_family == "hematologic_solid_tumor":
                    disease_semantic_subprofile = "hematologic_solid_overlap_non_malignant"
                    disease_semantic_note = "hematologic solid-tumor-like semantics are present, but the broader blood / immune frame still looks mixed and non-malignant-dominant"
                elif voted_disease_family == "healthy_control":
                    disease_semantic_subprofile = "hematologic_non_malignant_overlap_immune_activation"
                    disease_semantic_note = "hematologic non-malignant semantics are present, but they remain mixed under immune-heavy activation rather than a clean healthy-control frame"
                elif voted_disease_family == "hematologic_malignancy":
                    disease_semantic_subprofile = "hematologic_malignancy_overlap_generic"
                    disease_semantic_note = "hematologic malignancy-like semantics are present, but the broader hematologic frame remains mixed and not stably malignant"
                else:
                    disease_semantic_subprofile = "hematologic_overlap_generic"
                    disease_semantic_note = "hematologic disease semantics are present, but they remain mixed under broader immune or context tension"
            elif voted_site_family == "cutaneous":
                if "immune" in anchor_context or "infiltrated" in anchor_tumor.lower():
                    disease_semantic_subprofile = "cutaneous_overlap_immune_activation"
                    disease_semantic_note = "cutaneous tumor-like semantics are present, but the broader frame looks immune-heavy rather than stably cutaneous-malignant"
                else:
                    disease_semantic_subprofile = "cutaneous_overlap_generic"
                    disease_semantic_note = "cutaneous tumor-like semantics are present, but disease support remains mixed"
            elif voted_site_family == "genitourinary":
                if anchor_context == "unsupported" or "uncertain or activated non-malignant context" in anchor_tumor.lower():
                    disease_semantic_subprofile = "genitourinary_overlap_unsupported_context"
                    disease_semantic_note = "genitourinary tumor-like semantics are present, but the broader context is too unsupported or non-malignant-leaning to stabilize them"
                else:
                    disease_semantic_subprofile = "genitourinary_overlap_generic"
                    disease_semantic_note = "genitourinary tumor-like semantics are present, but disease support remains mixed"
            else:
                disease_semantic_subprofile = "mixed_semantic_generic"
    if disease_semantic_status == "stable":
        disease_what_to_trust = "the resolved disease-family semantic, but only at broad family granularity"
        disease_what_not_to_overcall = "fine disease naming beyond the current semantic family"
        disease_biological_readout = "disease-family semantics are stable enough to use as a broad disease layer"
        if resolved_disease_family == "other_solid_tumor":
            if anchor_context == "thoracic_infiltrated_tumor_context":
                disease_semantic_subprofile = "other_solid_stable_thoracic_infiltrated"
                disease_semantic_note = "other_solid_tumor semantics are stable and reinforced by a thoracic infiltrated tumor anchor"
                disease_what_to_trust = "stable broad solid-tumor semantics with thoracic infiltrated tumor context"
                disease_what_not_to_overcall = "fine organ-level or histology-level tumor naming beyond the current broad solid-tumor frame"
                disease_biological_readout = "the disease layer is stably tumor-like, but the broader tumor context is infiltrated rather than histologically specific"
            elif anchor_context == "immune_heavy_tumor_context":
                disease_semantic_subprofile = "other_solid_stable_immune_heavy"
                disease_semantic_note = "other_solid_tumor semantics are stable under an immune-heavy tumor context"
                disease_what_to_trust = "stable broad solid-tumor semantics under immune-heavy tumor biology"
                disease_what_not_to_overcall = "fine disease naming beyond broad tumor-like interpretation"
                disease_biological_readout = "the disease layer is stably tumor-like, but the surrounding context is immune-heavy"
            else:
                disease_semantic_subprofile = "other_solid_stable_clean_tumor"
                disease_semantic_note = "other_solid_tumor semantics are stable and align with a cleaner tumor context"
                disease_what_to_trust = "stable broad solid-tumor semantics"
                disease_what_not_to_overcall = "specific tumor lineage beyond the current broad solid-tumor frame"
                disease_biological_readout = "the disease layer is stably tumor-like without a strong non-malignant override"
        elif resolved_disease_family == "digestive_tumor":
            if anchor_context == "immune_heavy_tumor_context":
                disease_semantic_subprofile = "digestive_stable_immune_heavy"
                disease_semantic_note = "digestive tumor semantics are stable, but they sit inside an immune-heavy tumor context rather than a clean digestive tumor frame"
                disease_what_to_trust = "stable broad digestive tumor semantics under immune-heavy tumor biology"
                disease_what_not_to_overcall = "fine digestive tumor lineage or histology beyond the current broad digestive tumor frame"
                disease_biological_readout = "the disease layer is stably digestive-tumor-like, but the surrounding transcriptomic frame is immune-heavy"
            else:
                disease_semantic_subprofile = "digestive_stable_clean_context"
                disease_semantic_note = "digestive tumor semantics are stable, but the anchor remains cleaner or less malignant than a definitive digestive tumor state"
                disease_what_to_trust = "stable broad digestive tumor semantics"
                disease_what_not_to_overcall = "specific digestive tumor naming without stronger malignant support"
                disease_biological_readout = "the disease layer is stably digestive-tumor-like, but the anchor remains cleaner than a fully committed malignant digestive frame"
        elif resolved_disease_family == "healthy_control":
            if anchor_context == "activated_non_malignant_immune_context":
                disease_semantic_subprofile = "healthy_control_stable_activated"
                disease_semantic_note = "healthy-control-like semantics are stable, but they sit on top of mild non-malignant immune activation"
                disease_what_to_trust = "stable baseline-like non-malignant semantics"
                disease_what_not_to_overcall = "perfectly clean healthy baseline without activation"
                disease_biological_readout = "the disease layer stays non-malignant and baseline-like, with some activation still present in the broader context"
            else:
                disease_semantic_subprofile = "healthy_control_stable_clean"
                disease_semantic_note = "healthy-control-like semantics are stable and consistent with a cleaner non-malignant frame"
                disease_what_to_trust = "stable healthy-control-like semantics"
                disease_what_not_to_overcall = "finer claims beyond broad baseline non-malignant biology"
                disease_biological_readout = "the disease layer reads as stable baseline non-malignant biology"
        elif resolved_disease_family == "genitourinary_non_tumor":
            disease_semantic_subprofile = "genitourinary_non_tumor_stable_clean"
            disease_semantic_note = "genitourinary non-tumor semantics are stable and align with a cleaner non-malignant anchor"
            disease_what_to_trust = "stable broad genitourinary non-malignant semantics"
            disease_what_not_to_overcall = "fine genitourinary disease naming beyond a broad non-tumor frame"
            disease_biological_readout = "the disease layer is stably genitourinary and non-malignant"
        elif resolved_disease_family == "infectious":
            if "uncertain or activated non-malignant context" in anchor_tumor.lower() or anchor_context in {"immune_heavy_tumor_context", "thoracic_infiltrated_tumor_context", "clean_context"}:
                disease_semantic_status = "mixed"
                disease_semantic_subprofile = "infectious_overlap_non_malignant_activation"
                disease_semantic_note = "infectious disease semantics are present, but the broader anchor remains too non-malignant or context-shifted to treat them as a stable infectious disease frame"
                disease_what_to_trust = "infectious-like overlap under non-malignant or context-shifted activation"
                disease_what_not_to_overcall = "stable infectious disease semantics"
                disease_biological_readout = "infectious-like disease semantics are present, but the broader context remains mixed and not stably infectious"
            else:
                disease_semantic_subprofile = "infectious_stable"
                disease_semantic_note = "infectious disease semantics are stable at broad family level"
                disease_what_to_trust = "stable broad infectious disease semantics"
                disease_what_not_to_overcall = "specific infectious etiology without stronger support"
                disease_biological_readout = "the disease layer is stably infectious at broad family granularity"
        elif resolved_disease_family == "hematologic_malignancy":
            disease_semantic_subprofile = "hematologic_malignancy_stable"
            disease_semantic_note = "hematologic malignancy semantics are stable at broad family level"
            disease_what_to_trust = "stable broad hematologic malignancy semantics"
            disease_what_not_to_overcall = "specific leukemia or lymphoma subtype"
            disease_biological_readout = "the disease layer is stably hematologic-malignancy-like at broad family granularity"
        elif resolved_disease_family == "hematologic_solid_tumor":
            disease_semantic_subprofile = "hematologic_solid_stable_immune_heavy"
            disease_semantic_note = "hematologic solid-tumor-like semantics are stable, but the broader context remains immune-heavy"
            disease_what_to_trust = "stable broad hematologic tumor-like semantics under immune-heavy context"
            disease_what_not_to_overcall = "specific hematologic solid-tumor identity"
            disease_biological_readout = "the disease layer is stably hematologic tumor-like, but the surrounding context remains immune-heavy"
        elif resolved_disease_family == "cutaneous_tumor":
            disease_semantic_subprofile = "cutaneous_stable_unsupported"
            disease_semantic_note = "cutaneous tumor-like semantics are stable, but the structured context remains weak or unsupported"
            disease_what_to_trust = "stable broad cutaneous tumor-like semantics with caution"
            disease_what_not_to_overcall = "specific cutaneous malignancy identity under unsupported context"
            disease_biological_readout = "the disease layer is stably cutaneous tumor-like, but the broader context is too weak to over-interpret"
        elif resolved_disease_family == "other_non_tumor":
            disease_semantic_subprofile = "other_non_tumor_stable_immune_heavy"
            disease_semantic_note = "other non-tumor semantics are stable, but the broader frame remains immune-heavy and activated"
            disease_what_to_trust = "stable broad non-tumor semantics"
            disease_what_not_to_overcall = "clean baseline non-tumor biology without context tension"
            disease_biological_readout = "the disease layer stays non-tumor, but the surrounding context remains immune-heavy rather than clean baseline"
    elif disease_semantic_status == "mixed":
        disease_what_to_trust = "the resolved mixed disease semantic as an overlap state, not a final label"
        disease_what_not_to_overcall = "a hard disease-family diagnosis from semantic retrieval alone"
        disease_biological_readout = "treat the disease layer as mixed overlap anchored by the broader state family"
        if disease_semantic_subprofile == "hematologic_solid_overlap_non_malignant":
            disease_what_to_trust = "hematologic overlap under non-malignant immune dominance"
            disease_what_not_to_overcall = "stable hematologic solid-tumor disease identity"
            disease_biological_readout = "blood or marrow semantics are tumor-leaning, but the broader hematologic frame still reads as non-malignant-leaning activation"
        elif disease_semantic_subprofile == "other_solid_overlap_clean_anchor":
            disease_what_to_trust = "other-solid-tumor overlap only at broad semantic level"
            disease_what_not_to_overcall = "stable other_solid_tumor identity against a clean anchor"
            disease_biological_readout = "solid-tumor-like semantics are present, but the structured anchor stays cleaner than a stable malignant solid-tumor frame"
        elif disease_semantic_subprofile == "other_solid_overlap_non_malignant_activation":
            disease_what_to_trust = "other-solid-tumor overlap under non-malignant activation"
            disease_what_not_to_overcall = "stable solid-tumor disease semantics in the face of activated non-malignant context"
            disease_biological_readout = "solid-tumor-like semantics are present, but activated non-malignant biology still dominates the broader frame"
        elif disease_semantic_subprofile == "other_solid_overlap_immune_infiltrated":
            disease_what_to_trust = "other-solid-tumor overlap under infiltrated or immune-heavy tumor context"
            disease_what_not_to_overcall = "a cleaner standalone solid-tumor disease label"
            disease_biological_readout = "solid-tumor-like disease semantics are present, but the frame is strongly infiltrated or immune-heavy"
        elif disease_semantic_subprofile == "other_solid_overlap_generic":
            disease_what_to_trust = "only a broad other-solid-tumor overlap"
            disease_what_not_to_overcall = "stable or lineage-specific solid-tumor semantics"
            disease_biological_readout = "the disease layer leans solid-tumor-like, but the surrounding context remains too mixed to stabilize it"
        elif disease_semantic_subprofile == "hematologic_non_malignant_overlap_immune_activation":
            disease_what_to_trust = "hematologic non-malignant overlap with immune-heavy activation"
            disease_what_not_to_overcall = "clean healthy-control semantics"
            disease_biological_readout = "the disease layer stays hematologic and non-malignant-leaning, but activation is too strong to read as clean baseline blood biology"
        elif disease_semantic_subprofile == "hematologic_malignancy_overlap_infiltrated":
            disease_what_to_trust = "hematologic malignancy-like overlap under infiltrated or immune-heavy context"
            disease_what_not_to_overcall = "stable leukemia- or lymphoma-like disease identity"
            disease_biological_readout = "hematologic malignancy-like semantics are present, but infiltrated or immune-heavy context prevents a stable malignant disease interpretation"
        elif disease_semantic_subprofile == "hematologic_malignancy_overlap_generic":
            disease_what_to_trust = "hematologic malignancy-like overlap only at broad semantic level"
            disease_what_not_to_overcall = "a stable leukemia- or lymphoma-like label"
            disease_biological_readout = "malignancy-like hematologic semantics are present, but the supporting frame remains mixed"
        elif disease_semantic_subprofile == "hematologic_malignancy_overlap_non_malignant":
            disease_what_to_trust = "hematologic malignancy-like overlap under non-malignant immune override"
            disease_what_not_to_overcall = "stable malignant hematologic disease semantics"
            disease_biological_readout = "malignancy-like hematologic semantics are present, but the broader blood or immune frame still reads more non-malignant than stably malignant"
        elif disease_semantic_subprofile == "cutaneous_overlap_immune_activation":
            disease_what_to_trust = "cutaneous tumor-like overlap under immune-heavy activation"
            disease_what_not_to_overcall = "stable cutaneous tumor disease semantics"
            disease_biological_readout = "cutaneous tumor-leaning semantics are present, but immune-heavy context dominates the broader frame"
        elif disease_semantic_subprofile == "cutaneous_overlap_generic":
            disease_what_to_trust = "only a cutaneous tumor-like overlap"
            disease_what_not_to_overcall = "stable cutaneous tumor semantics without stronger support"
            disease_biological_readout = "cutaneous tumor-like disease semantics are present, but the broader frame remains mixed and insufficiently specific"
        elif disease_semantic_subprofile == "digestive_overlap_clean_anchor":
            disease_what_to_trust = "only a digestive tumor-like overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable digestive tumor semantics"
            disease_biological_readout = "digestive tumor-like disease semantics are present, but the cleaner anchor keeps them from stabilizing as a definitive digestive tumor state"
        elif disease_semantic_subprofile == "digestive_overlap_non_malignant_activation":
            disease_what_to_trust = "digestive tumor-like overlap under activated non-malignant context"
            disease_what_not_to_overcall = "stable digestive tumor disease semantics"
            disease_biological_readout = "digestive tumor-like disease semantics are present, but activated non-malignant biology dominates the broader frame"
        elif disease_semantic_subprofile == "digestive_overlap_immune_infiltrated":
            disease_what_to_trust = "digestive tumor-like overlap under immune-infiltrated context"
            disease_what_not_to_overcall = "stable digestive tumor semantics"
            disease_biological_readout = "digestive tumor-like disease semantics are present, but they are embedded in immune-heavy or infiltrated biology rather than a clean digestive tumor frame"
        elif disease_semantic_subprofile == "digestive_overlap_generic":
            disease_what_to_trust = "only a broad digestive tumor-like overlap"
            disease_what_not_to_overcall = "stable digestive tumor semantics without stronger support"
            disease_biological_readout = "digestive tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "breast_overlap_clean_anchor":
            disease_what_to_trust = "only a breast-tumor-like overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable breast tumor semantics"
            disease_biological_readout = "breast-tumor-like disease semantics are present, but the cleaner anchor keeps them from stabilizing"
        elif disease_semantic_subprofile == "breast_overlap_non_malignant_activation":
            disease_what_to_trust = "breast-tumor-like overlap under non-malignant activation"
            disease_what_not_to_overcall = "stable breast tumor semantics"
            disease_biological_readout = "breast-tumor-like disease semantics are present, but activated non-malignant biology dominates the broader frame"
        elif disease_semantic_subprofile == "breast_overlap_immune_infiltrated":
            disease_what_to_trust = "breast-tumor-like overlap under infiltrated context"
            disease_what_not_to_overcall = "stable breast tumor semantics"
            disease_biological_readout = "breast-tumor-like disease semantics are present, but they are embedded in infiltrated or immune-heavy biology"
        elif disease_semantic_subprofile == "breast_overlap_generic":
            disease_what_to_trust = "only a broad breast-tumor-like overlap"
            disease_what_not_to_overcall = "stable breast tumor semantics without stronger support"
            disease_biological_readout = "breast-tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "breast_non_malignant_overlap_clean_anchor":
            disease_what_to_trust = "only a breast non-malignant overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable breast non-malignant semantics"
            disease_biological_readout = "breast non-malignant disease semantics are present, but the anchor is cleaner than the semantic frame"
        elif disease_semantic_subprofile == "breast_non_malignant_overlap_non_malignant_activation":
            disease_what_to_trust = "breast non-malignant overlap under activated non-malignant context"
            disease_what_not_to_overcall = "stable breast non-malignant semantics"
            disease_biological_readout = "breast non-malignant semantics are present, but broader activation keeps them mixed"
        elif disease_semantic_subprofile == "breast_non_malignant_overlap_immune_infiltrated":
            disease_what_to_trust = "breast non-malignant overlap under infiltrated context"
            disease_what_not_to_overcall = "stable breast non-malignant semantics"
            disease_biological_readout = "breast non-malignant semantics are present, but the broader frame is infiltrated or context-heavy"
        elif disease_semantic_subprofile == "breast_non_malignant_overlap_generic":
            disease_what_to_trust = "only a broad breast non-malignant overlap"
            disease_what_not_to_overcall = "stable breast non-malignant semantics"
            disease_biological_readout = "breast non-malignant disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "thoracic_overlap_clean_anchor":
            disease_what_to_trust = "only a thoracic-tumor-like overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable thoracic tumor semantics"
            disease_biological_readout = "thoracic-tumor-like disease semantics are present, but the cleaner anchor keeps them from stabilizing"
        elif disease_semantic_subprofile == "thoracic_overlap_non_malignant_activation":
            disease_what_to_trust = "thoracic-tumor-like overlap under non-malignant activation"
            disease_what_not_to_overcall = "stable thoracic tumor semantics"
            disease_biological_readout = "thoracic-tumor-like disease semantics are present, but the broader context remains non-malignant-leaning or uncertain"
        elif disease_semantic_subprofile == "thoracic_overlap_immune_infiltrated":
            disease_what_to_trust = "thoracic-tumor-like overlap under infiltrated context"
            disease_what_not_to_overcall = "stable thoracic tumor semantics"
            disease_biological_readout = "thoracic-tumor-like disease semantics are present, but they sit inside infiltrated or immune-heavy tumor biology"
        elif disease_semantic_subprofile == "thoracic_overlap_generic":
            disease_what_to_trust = "only a broad thoracic-tumor-like overlap"
            disease_what_not_to_overcall = "stable thoracic tumor semantics"
            disease_biological_readout = "thoracic-tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "digestive_solid_overlap_clean_anchor":
            disease_what_to_trust = "only a digestive-solid-tumor-like overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable digestive solid-tumor semantics"
            disease_biological_readout = "digestive solid-tumor-like disease semantics are present, but the cleaner anchor prevents a stable malignant readout"
        elif disease_semantic_subprofile == "digestive_solid_overlap_non_malignant_activation":
            disease_what_to_trust = "digestive-solid-tumor-like overlap under non-malignant activation"
            disease_what_not_to_overcall = "stable digestive solid-tumor semantics"
            disease_biological_readout = "digestive solid-tumor-like disease semantics are present, but non-malignant activation dominates the broader frame"
        elif disease_semantic_subprofile == "digestive_solid_overlap_immune_infiltrated":
            disease_what_to_trust = "digestive-solid-tumor-like overlap under infiltrated context"
            disease_what_not_to_overcall = "stable digestive solid-tumor semantics"
            disease_biological_readout = "digestive solid-tumor-like disease semantics are present, but infiltrated or immune-heavy context prevents a stable call"
        elif disease_semantic_subprofile == "digestive_solid_overlap_generic":
            disease_what_to_trust = "only a broad digestive-solid-tumor-like overlap"
            disease_what_not_to_overcall = "stable digestive solid-tumor semantics"
            disease_biological_readout = "digestive solid-tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "cns_overlap_clean_anchor":
            disease_what_to_trust = "only a cns-tumor-like overlap against a cleaner anchor"
            disease_what_not_to_overcall = "stable CNS tumor semantics"
            disease_biological_readout = "CNS-tumor-like disease semantics are present, but the cleaner anchor prevents stable interpretation"
        elif disease_semantic_subprofile == "cns_overlap_non_malignant_activation":
            disease_what_to_trust = "CNS-tumor-like overlap under non-malignant activation"
            disease_what_not_to_overcall = "stable CNS tumor semantics"
            disease_biological_readout = "CNS-tumor-like disease semantics are present, but activated non-malignant biology dominates the broader frame"
        elif disease_semantic_subprofile == "cns_overlap_immune_infiltrated":
            disease_what_to_trust = "CNS-tumor-like overlap under infiltrated context"
            disease_what_not_to_overcall = "stable CNS tumor semantics"
            disease_biological_readout = "CNS-tumor-like disease semantics are present, but they are embedded in infiltrated or immune-heavy context"
        elif disease_semantic_subprofile == "cns_overlap_generic":
            disease_what_to_trust = "only a broad CNS-tumor-like overlap"
            disease_what_not_to_overcall = "stable CNS tumor semantics"
            disease_biological_readout = "CNS-tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "genitourinary_overlap_unsupported_context":
            disease_what_to_trust = "only a genitourinary tumor-like overlap under unsupported context"
            disease_what_not_to_overcall = "stable genitourinary tumor semantics"
            disease_biological_readout = "genitourinary tumor-like disease semantics are present, but the broader context is too unsupported or non-malignant-leaning to stabilize them"
        elif disease_semantic_subprofile == "genitourinary_overlap_generic":
            disease_what_to_trust = "only a broad genitourinary tumor-like overlap"
            disease_what_not_to_overcall = "stable genitourinary tumor semantics without stronger support"
            disease_biological_readout = "genitourinary tumor-like disease semantics are present, but the broader frame remains mixed"
        elif disease_semantic_subprofile == "healthy_control_overlap_immune_activation":
            disease_what_to_trust = "healthy-control-like overlap only as a weak non-malignant baseline tendency"
            disease_what_not_to_overcall = "a clean healthy baseline state"
            disease_biological_readout = "baseline-like non-malignant semantics are present, but immune-heavy activation dominates the anchor"
        elif disease_semantic_subprofile == "healthy_control_overlap_unsupported_context":
            disease_what_to_trust = "only a weak healthy-control-like overlap"
            disease_what_not_to_overcall = "stable healthy-control semantics under unsupported context"
            disease_biological_readout = "the semantic disease layer leans non-malignant baseline, but the broader context is too unsupported to treat it as stable"
        elif disease_semantic_subprofile == "hematologic_solid_overlap_unsupported_context":
            disease_what_to_trust = "only a hematologic solid-tumor-like overlap under unsupported context"
            disease_what_not_to_overcall = "stable hematologic solid-tumor semantics"
            disease_biological_readout = "hematologic tumor-leaning disease semantics are present, but the structured context is too unsupported to stabilize them"
    else:
        disease_what_to_trust = "only that disease-level semantics are unsupported or unstable"
        disease_what_not_to_overcall = "any disease-family interpretation"
        disease_biological_readout = "use site/context/state, not the disease layer"
    hematologic_subprofile = "none"
    epithelial_subprofile = "none"
    clean_anchor_subprofile = "none"
    generic_subprofile = "none"
    if explanation_mode == "context_override_explainer":
        if voted_site_family == "hematologic":
            mixed_override_profile = "hematologic_tumor_like_vs_non_malignant_immune"
        elif anchor_context == "clean_context":
            mixed_override_profile = "tumor_like_semantics_vs_clean_anchor"
        elif "hematologic" in anchor_tissue or "blood" in anchor_tissue or "immune" in anchor_tissue:
            mixed_override_profile = "epithelial_or_other_tumor_like_vs_immune_dominant_anchor"
        else:
            mixed_override_profile = "generic_context_override"
    elif explanation_mode == "family_conflict_explainer":
        mixed_override_profile = "prototype_neighbor_family_conflict"
    else:
        mixed_override_profile = "none"

    if mixed_override_profile == "hematologic_tumor_like_vs_non_malignant_immune":
        if voted_disease_family == "hematologic_malignancy":
            hematologic_subprofile = "hematologic_malignancy_like_immune_activation"
            primary = (
                "hematologic malignancy-like semantics are present, but the broader blood / immune transcriptomic frame still looks more activated and non-malignant than a stable leukemia- or lymphoma-like state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the sample reads more like immune-activated hematologic biology with malignancy-like overlap than a hard hematologic malignancy call"
            )
            profile_summary = (
                "hematologic malignancy-like semantics are present, but the broader blood / immune context still looks more activated and non-malignant than a stable leukemia- or lymphoma-like frame"
            )
            profile_recommendation = (
                "treat this as immune-activated hematologic biology with malignancy-like overlap; keep hematologic malignancy as a semantic possibility, but do not collapse it into a hard malignant call"
            )
        else:
            hematologic_subprofile = "hematologic_non_malignant_like_immune_activation"
            primary = (
                "hematologic immune activation is dominant, and any malignancy-like semantics are weaker than the broader non-malignant blood / immune activation frame"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the sample is better explained as non-malignant hematologic immune activation with some tumor-like overlap"
            )
            profile_summary = (
                "hematologic immune activation is dominant, and the broader blood / immune context stays non-malignant; any malignancy-like semantics appear weaker than the immune activation frame"
            )
            profile_recommendation = (
                "treat this as non-malignant hematologic immune activation with some tumor-like overlap, not as a stable hematologic malignancy interpretation"
            )
    elif mixed_override_profile == "epithelial_or_other_tumor_like_vs_immune_dominant_anchor":
        if "immune-infiltrated tumor-like" in anchor_tumor.lower():
            epithelial_subprofile = "epithelial_tumor_like_immune_infiltrated_activation"
            primary = (
                "epithelial or other solid-tumor-like semantics are present, but the surrounding transcriptomic frame is heavily immune-dominant and better explained as infiltrated mixed biology than a stable epithelial tumor state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue} and already reads as immune-infiltrated tumor-like context; "
                "the semantic tumor frame is plausible, but it is inseparable from strong immune dominance"
            )
            profile_summary = (
                "epithelial or other solid-tumor-like semantics are present, but they sit inside an already immune-infiltrated tumor-like context rather than a clean solid-tumor frame"
            )
            profile_recommendation = (
                "treat this as immune-infiltrated epithelial-like mixed biology; prioritize infiltration and context over a hard solid-tumor label"
            )
        else:
            epithelial_subprofile = "epithelial_tumor_like_non_malignant_activation"
            primary = (
                "epithelial or other solid-tumor-like semantics are present, but the broader transcriptomic context still looks more like activated non-malignant immune biology than a stable epithelial tumor state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the sample is better explained as immune-activated mixed biology with epithelial tumor-like overlap"
            )
            profile_summary = (
                "epithelial or other solid-tumor-like semantics are present, but the surrounding transcriptomic context is dominated by activated non-malignant immune / hematologic features"
            )
            profile_recommendation = (
                "treat this as immune-activated mixed biology with epithelial tumor-like overlap rather than a stable solid-tumor call"
            )
    elif mixed_override_profile == "tumor_like_semantics_vs_clean_anchor":
        if "immune-infiltrated tumor-like" in anchor_tumor.lower():
            clean_anchor_subprofile = "tumor_like_clean_anchor_infiltrated"
            primary = (
                "tumor-like semantics are present, but the structured anchor stays cleaner than the semantic tumor frame and reads more like an infiltrated or context-heavy sample than a stable malignant state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue} while still looking cleaner than the semantic tumor frame; "
                "this is better treated as an infiltrated mixed state than as a stable solid-tumor interpretation"
            )
            profile_summary = (
                "tumor-like semantics are present, but the cleaner anchor still suggests an infiltrated or context-heavy sample rather than a stable malignant frame"
            )
            profile_recommendation = (
                "treat this as a context-heavy infiltrated mixed state with tumor-like overlap, not as a stable malignant interpretation"
            )
        else:
            clean_anchor_subprofile = "tumor_like_clean_anchor_weak"
            primary = (
                "tumor-like semantics are present, but the structured anchor remains cleaner and less malignant than the semantic tumor frame"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the tumor-like signal is weaker than the cleaner background context"
            )
            profile_summary = (
                "tumor-like semantics are present, but the structured anchor remains cleaner and less malignant than the semantic tumor frame"
            )
            profile_recommendation = (
                "treat this as a weak tumor-like hypothesis under a cleaner background, not a stable malignant interpretation"
            )
    elif mixed_override_profile == "prototype_neighbor_family_conflict":
        profile_summary = "prototype and semantic neighbors disagree at the broad tissue-family level"
        profile_recommendation = "inspect competing tissue families before trusting broad disease semantics"
    elif mixed_override_profile == "generic_context_override":
        if anchor_context == "immune_heavy_tumor_context":
            generic_subprofile = "generic_tumor_like_context_heavy"
            primary = (
                "tumor-like semantics are present, but the surrounding transcriptomic frame is context-heavy rather than a stable tissue-of-origin-specific malignant state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the sample is better read as context-heavy tumor-associated mixed biology than as a clean tumor semantic frame"
            )
            profile_summary = (
                "tumor-like semantics are present, but they are embedded in a context-heavy frame that overwhelms a stable tissue-specific semantic call"
            )
            profile_recommendation = (
                "treat this as context-heavy tumor-associated mixed biology; prioritize the broader context over a hard tissue-of-origin-specific tumor label"
            )
        else:
            generic_subprofile = "generic_tumor_like_non_malignant_context"
            primary = (
                "tumor-like semantics are present, but the broader transcriptomic context still looks more like activated or non-malignant mixed biology than a stable malignant state"
            )
            counter = (
                f"structured anchor points to {anchor_context} with {anchor_tissue}; "
                "the sample is better read as activated non-malignant mixed biology with tumor-like overlap"
            )
            profile_summary = (
                "tumor-like semantics are present, but the surrounding transcriptomic context is dominated by activated non-malignant mixed biology"
            )
            profile_recommendation = (
                "treat this as activated non-malignant mixed biology with tumor-like overlap, not as a stable malignant interpretation"
            )
    else:
        hematologic_subprofile = "none"
        profile_summary = "no special mixed-override profile"
        profile_recommendation = "broad semantic frame is usable"

    if disable_state_map:
        if explanation_mode == "stable_consensus":
            state_family = "stable_consensus"
            state_subprofile = "stable_consensus"
        elif explanation_mode == "unsupported_semantics":
            state_family = "unsupported_semantics"
            state_subprofile = "unsupported_semantics"
        elif explanation_mode == "family_conflict_explainer":
            state_family = "mixed_semantic_generic"
            state_subprofile = "prototype_neighbor_family_conflict"
        else:
            state_family = "mixed_semantic_generic"
            state_subprofile = "mixed_semantic_generic"
            profile_summary = "mixed tumor-like semantics remain unresolved without semantic state-map organization"
            profile_recommendation = "use broad semantic direction only; mixed-biology tension is not explicitly organized in this ablation"
    elif explanation_mode == "stable_consensus":
        state_family = "stable_consensus"
        state_subprofile = "stable_consensus"
    elif explanation_mode == "unsupported_semantics":
        state_family = "unsupported_semantics"
        state_subprofile = "unsupported_semantics"
    elif mixed_override_profile == "hematologic_tumor_like_vs_non_malignant_immune":
        state_family = "hematologic_override"
        state_subprofile = hematologic_subprofile
    elif mixed_override_profile == "epithelial_or_other_tumor_like_vs_immune_dominant_anchor":
        state_family = "epithelial_override"
        state_subprofile = epithelial_subprofile
    elif mixed_override_profile == "tumor_like_semantics_vs_clean_anchor":
        state_family = "clean_anchor_override"
        state_subprofile = clean_anchor_subprofile
    elif mixed_override_profile == "generic_context_override":
        state_family = "generic_context_override"
        state_subprofile = generic_subprofile
    elif mixed_override_profile == "prototype_neighbor_family_conflict":
        state_family = "family_conflict"
        state_subprofile = "prototype_neighbor_family_conflict"
    else:
        state_family = "other"
        state_subprofile = "other"

    if state_family == "stable_consensus":
        what_to_trust = "semantic majority and prototype-neighbor agreement at broad family level"
        what_not_to_overcall = "fine disease wording beyond the broad semantic family"
        biological_readout = "treat the current semantic family as the main biological frame"
    elif state_family in {"hematologic_override", "epithelial_override", "clean_anchor_override", "generic_context_override"}:
        what_to_trust = "the override state itself and the main semantic-vs-anchor tension"
        what_not_to_overcall = "a hard disease-family call from the semantic side alone"
        biological_readout = "prioritize the mixed-state interpretation over the raw disease-family tag"
    elif state_family == "unsupported_semantics":
        what_to_trust = "only the instability itself"
        what_not_to_overcall = "any disease-family interpretation"
        biological_readout = "treat this sample as open-world / unsupported at disease level"
    else:
        what_to_trust = "broad semantic direction only"
        what_not_to_overcall = "fine disease-family semantics"
        biological_readout = "use the state card as orientation, not as a final disease interpretation"

    evidence_highlights: list[dict[str, str]] = [
        {
            "kind": "semantic_majority",
            "value": f"{voted_site_family} / {voted_disease_family}",
            "strength": semantic_strength,
        },
        {
            "kind": "anchor_context",
            "value": f"{anchor_context} ({anchor_tissue})",
            "strength": anchor_support_conf,
        },
    ]
    if dominant_neighbor_tags:
        evidence_highlights.append(
            {
                "kind": "neighbor_context",
                "value": ", ".join(dominant_neighbor_tags[:3]),
                "strength": dominant_tag_strength,
            }
        )
    if age_driver_overlap:
        evidence_highlights.append(
            {
                "kind": "age_driver_overlap",
                "value": ", ".join(age_driver_overlap[:3]),
                "strength": age_overlap_strength,
            }
        )
    if bool(semantic_consensus.get("anchor_adjusted", False)):
        evidence_highlights.append(
            {
                "kind": "anchor_override",
                "value": "semantic frame was downgraded by structured anchor evidence",
                "strength": 1.0,
            }
        )
    evidence_highlights.append(
        {
            "kind": "tension",
            "value": f"{voted_site_family} / {voted_disease_family} vs {anchor_context}",
            "strength": 1.0 if bool(semantic_consensus.get("anchor_adjusted", False)) else float(max(0.25, abs(semantic_strength - anchor_support_conf))),
        }
    )
    ranked_evidence = _rank_state_evidence(
        state_family,
        state_subprofile,
        str(semantic_consensus.get("route", "unknown")),
        explanation_mode,
        bool(semantic_consensus.get("anchor_adjusted", False)),
        evidence_highlights,
    )

    disable_portrait = os.getenv("SEMANTIC_DISABLE_PORTRAIT", "0") == "1"

    if disease_semantic_status == "stable":
        portrait_confidence = "stable broad semantic frame"
    elif disease_semantic_status == "mixed":
        portrait_confidence = "mixed but interpretable semantic frame"
    else:
        portrait_confidence = "unsupported disease-level semantic frame"

    disease_phrase = str(resolved_disease_family or voted_disease_family or "unknown").replace("_", " ")
    state_phrase = str(state_subprofile or state_family or "unknown").replace("_", " ")
    context_phrase = str(anchor_context or "unknown").replace("_", " ")
    tissue_phrase = str(anchor_tissue or voted_site_family or "unknown").replace("_", " ")
    age_years = age_like_state.get("years")
    age_phrase = f"{age_years:.1f} years ({age_band.replace('_', ' ')})" if age_years is not None else age_band.replace("_", " ")

    if disease_semantic_status == "stable":
        portrait_headline = f"Broad disease semantics are stable around {disease_phrase}."
    elif disease_semantic_status == "mixed":
        portrait_headline = f"The RNA reads as a mixed open-world state centered on {disease_phrase} overlap."
    else:
        portrait_headline = "Disease-level semantics are not stable enough to trust directly."

    portrait_biology = (
        f"Current state: {state_phrase}. "
        f"Anchor context: {context_phrase} in {tissue_phrase} biology. "
        f"Transcriptomic age-like signal: {age_phrase}."
    )
    portrait_caution = (
        disease_what_not_to_overcall
        if disease_what_not_to_overcall
        else what_not_to_overcall
    )
    portrait_next = (
        profile_recommendation
        if profile_recommendation and profile_recommendation != "broad semantic frame is usable"
        else disease_biological_readout
    )

    return {
        "mode": explanation_mode,
        "primary_hypothesis": primary,
        "counterevidence": counter,
        "prototype_semantic_family": {
            "site_family": proto_site_family,
            "disease_family": proto_disease_family,
        },
        "neighbor_semantic_majority": {
            "site_family": voted_site_family,
            "disease_family": voted_disease_family,
            "site_family_votes": semantic_consensus.get("site_family_distribution", {}),
            "disease_family_votes": semantic_consensus.get("disease_family_distribution", {}),
        },
        "disease_semantic_resolution": {
            "raw_disease_family": voted_disease_family,
            "resolved_disease_family": resolved_disease_family,
            "status": disease_semantic_status,
            "subprofile": disease_semantic_subprofile,
            "note": disease_semantic_note,
        },
        "disease_semantic_card": {
            "status": disease_semantic_status,
            "raw_disease_family": voted_disease_family,
            "resolved_disease_family": resolved_disease_family,
            "subprofile": disease_semantic_subprofile,
            "note": disease_semantic_note,
            "what_to_trust": disease_what_to_trust,
            "what_not_to_overcall": disease_what_not_to_overcall,
            "biological_readout": disease_biological_readout,
        },
        "anchor_context": {
            "support_level": anchor_support,
            "context_state": anchor_context,
            "tissue_context": anchor_tissue,
            "tumor_status": anchor_tumor,
        },
        "age_like_context": {
            "band": age_band,
            "possible_drivers": age_drivers,
        },
        "mixed_biology_evidence": {
            "override_profile": mixed_override_profile,
            "hematologic_subprofile": hematologic_subprofile,
            "epithelial_subprofile": epithelial_subprofile,
            "clean_anchor_subprofile": clean_anchor_subprofile,
            "generic_subprofile": generic_subprofile,
            "profile_summary": profile_summary,
            "profile_recommendation": profile_recommendation,
            "dominant_neighbor_context_tags": dominant_neighbor_tags,
            "age_driver_overlap": age_driver_overlap,
            "anchor_override_active": bool(semantic_consensus.get("anchor_adjusted", False)),
            "tissue_vs_context_tension": (
                f"semantic family points to {voted_site_family} / {voted_disease_family}, "
                f"while anchor context is {anchor_context} with {anchor_tissue}"
            ),
        },
        "semantic_state_card": {
            "state_family": state_family,
            "state_subprofile": state_subprofile,
            "state_summary": profile_summary,
            "state_recommendation": profile_recommendation,
            "state_mode": explanation_mode,
            "what_to_trust": what_to_trust,
            "what_not_to_overcall": what_not_to_overcall,
            "biological_readout": biological_readout,
            "evidence_highlights": ranked_evidence,
        },
        "semantic_portrait": (
            {}
            if disable_portrait
            else {
                "headline": portrait_headline,
                "confidence_posture": portrait_confidence,
                "current_state": state_phrase,
                "resolved_disease_semantic": disease_phrase,
                "anchor_context": context_phrase,
                "tissue_context": tissue_phrase,
                "age_like_years": age_years,
                "age_like_band": age_band,
                "biological_summary": portrait_biology,
                "what_to_trust": disease_what_to_trust or what_to_trust,
                "caution": portrait_caution,
                "recommended_reading": portrait_next,
            }
        ),
        "interpretation": {
            "likely_frame": (
                "context_or_non_malignant_override"
                if explanation_mode == "context_override_explainer"
                else ("prototype_neighbor_family_conflict" if explanation_mode == "family_conflict_explainer" else "stable_or_other")
            ),
            "recommended_focus": (
                profile_recommendation
                if explanation_mode in {"context_override_explainer", "family_conflict_explainer"}
                else "broad semantic frame is usable"
            ),
        },
        "neighbor_site_families": neighbor_sites,
        "neighbor_disease_families": neighbor_diseases,
    }


def _weighted_vote(items: list[str], weights: list[float]) -> tuple[str, float, dict[str, float]]:
    votes: dict[str, float] = {}
    for item, weight in zip(items, weights):
        key = str(item or "unknown")
        if key in {"", "nan", "None"}:
            key = "unknown"
        votes[key] = votes.get(key, 0.0) + float(max(weight, 0.0) + 1e-6)
    if not votes:
        return "unknown", 0.0, {}
    total = max(sum(votes.values()), 1e-6)
    best = max(votes, key=votes.get)
    dist = {k: float(v / total) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}
    return best, float(votes[best] / total), dist


def _rank_state_evidence(
    state_family: str,
    state_subprofile: str,
    semantic_route: str,
    explanation_mode: str,
    anchor_adjusted: bool,
    evidence_highlights: list[dict[str, object]],
) -> list[dict[str, object]]:
    priority_map = {
        "stable_consensus": {
            "semantic_majority": 1.00,
            "anchor_context": 0.90,
            "neighbor_context": 0.70,
            "tension": 0.55,
            "age_driver_overlap": 0.35,
            "anchor_override": 0.10,
        },
        "hematologic_override": {
            "anchor_context": 1.00,
            "semantic_majority": 0.95,
            "neighbor_context": 0.85,
            "age_driver_overlap": 0.75,
            "tension": 0.70,
            "anchor_override": 0.90,
        },
        "epithelial_override": {
            "anchor_context": 1.00,
            "neighbor_context": 0.92,
            "semantic_majority": 0.88,
            "tension": 0.78,
            "anchor_override": 0.90,
            "age_driver_overlap": 0.45,
        },
        "clean_anchor_override": {
            "anchor_context": 1.00,
            "tension": 0.92,
            "semantic_majority": 0.84,
            "neighbor_context": 0.72,
            "anchor_override": 0.88,
            "age_driver_overlap": 0.40,
        },
        "generic_context_override": {
            "anchor_context": 1.00,
            "neighbor_context": 0.90,
            "tension": 0.82,
            "semantic_majority": 0.80,
            "anchor_override": 0.88,
            "age_driver_overlap": 0.42,
        },
        "family_conflict": {
            "semantic_majority": 1.00,
            "tension": 0.95,
            "anchor_context": 0.75,
            "neighbor_context": 0.70,
            "anchor_override": 0.50,
            "age_driver_overlap": 0.20,
        },
        "unsupported_semantics": {
            "tension": 1.00,
            "anchor_context": 0.90,
            "semantic_majority": 0.50,
            "neighbor_context": 0.45,
            "anchor_override": 0.55,
            "age_driver_overlap": 0.25,
        },
        "other": {
            "semantic_majority": 0.90,
            "anchor_context": 0.85,
            "neighbor_context": 0.70,
            "tension": 0.65,
            "anchor_override": 0.50,
            "age_driver_overlap": 0.30,
        },
    }
    priorities = priority_map.get(state_family, priority_map["other"])
    priors = _load_state_evidence_priors()
    family_prior = priors.get("family_priors", {}).get(state_family, {})
    subprofile_prior = priors.get("subprofile_priors", {}).get(state_subprofile, {})
    model = _load_state_evidence_scorer()
    model_probs: dict[str, float] = {}
    if model is not None:
        try:
            model_probs = {
                str(label): float(prob)
                for label, prob in zip(
                    model.classes_,
                    model.predict_proba(
                        [
                            _state_evidence_model_features(
                                state_family,
                                state_subprofile,
                                semantic_route,
                                explanation_mode,
                                anchor_adjusted,
                                evidence_highlights,
                            )
                        ]
                    )[0],
                )
            }
        except Exception:
            model_probs = {}
    ranked = []
    for idx, item in enumerate(evidence_highlights):
        kind = str(item.get("kind", "other"))
        score = float(priorities.get(kind, 0.25))
        # Subprofile-specific adjustments to better reflect within-family biology tension.
        if "immune_heavy" in state_subprofile:
            if kind == "neighbor_context":
                score += 0.12
            elif kind == "tension":
                score += 0.08
        if "non_malignant" in state_subprofile:
            if kind == "anchor_context":
                score += 0.10
            elif kind == "anchor_override":
                score += 0.06
        if "infiltrated" in state_subprofile:
            if kind == "tension":
                score += 0.12
            elif kind == "anchor_context":
                score += 0.05
        if "malignancy_like" in state_subprofile:
            if kind == "semantic_majority":
                score += 0.08
            elif kind == "anchor_context":
                score += 0.04
        if "weak" in state_subprofile:
            if kind == "anchor_context":
                score += 0.08
            elif kind == "semantic_majority":
                score -= 0.06
        # Data-driven calibration from xlarge unseen validation.
        score += 0.22 * float(family_prior.get(kind, 0.0))
        score += 0.35 * float(subprofile_prior.get(kind, 0.0))
        score += 0.45 * float(model_probs.get(kind, 0.0))
        ranked.append(
            {
                **item,
                "base_score": score,
                "model_prob": float(model_probs.get(kind, 0.0)),
                "strength": float(item.get("strength", 1.0)),
                "score": score * (0.6 + 0.4 * float(item.get("strength", 1.0))),
                "rank": idx + 1,  # temporary; recomputed after sorting
            }
        )
    ranked.sort(key=lambda x: (-float(x["score"]), str(x.get("kind", ""))))
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i
    return ranked


def _build_semantic_consensus(top_prototype: dict, neighbors: list[dict]) -> dict:
    if not neighbors:
        return {
            "route": "unsupported_semantic_frame",
            "site_family": "unknown",
            "tumor_status": str(top_prototype.get("tumor_anchor", "unknown")),
            "disease_family": "unknown",
            "site_family_confidence": 0.0,
            "tumor_confidence": 0.0,
            "disease_family_confidence": 0.0,
            "reason": "no semantic neighbors available",
        }

    proto_site = str(top_prototype.get("site_anchor", "unknown"))
    proto_tumor = str(top_prototype.get("tumor_anchor", "unknown"))
    proto_disease = str(top_prototype.get("disease_anchor", "unknown"))
    proto_site_family = organ_family(proto_site)
    proto_disease_family = disease_family(proto_disease, proto_site, proto_tumor)

    n_weights = [float(n.get("score", 0.0)) for n in neighbors]
    n_site_family = [organ_family(str(n.get("site_anchor", "unknown"))) for n in neighbors]
    n_tumor = [str(n.get("tumor_anchor", "unknown")) for n in neighbors]
    n_disease_family = [
        disease_family(str(n.get("disease_anchor", "unknown")), str(n.get("site_anchor", "unknown")), str(n.get("tumor_anchor", "unknown")))
        for n in neighbors
    ]

    site_label, site_conf, site_dist = _weighted_vote([proto_site_family] + n_site_family, [1.0] + n_weights)
    tumor_label, tumor_conf, tumor_dist = _weighted_vote([proto_tumor] + n_tumor, [1.0] + n_weights)
    disease_label, disease_conf, disease_dist = _weighted_vote([proto_disease_family] + n_disease_family, [1.0] + n_weights)

    proto_matches_site = site_label == proto_site_family
    proto_matches_disease = disease_label == proto_disease_family
    stable = site_conf >= 0.60 and tumor_conf >= 0.75 and proto_matches_site
    mixed = tumor_conf >= 0.75
    route = "stable_semantic_frame" if stable else ("mixed_semantic_frame" if mixed else "unstable_semantic_frame")
    if route == "stable_semantic_frame":
        reason = "prototype and semantic neighbors agree at broad tissue/tumor level"
        route_subtype = "stable_broad_consensus"
    elif route == "mixed_semantic_frame":
        if not proto_matches_site:
            reason = "tumor-like semantics are stable, but prototype and neighbor tissue families disagree"
            route_subtype = "prototype_neighbor_family_conflict"
        elif not proto_matches_disease:
            reason = "tumor-like semantics are stable, but disease-family semantics remain mixed"
            route_subtype = "prototype_neighbor_disease_conflict"
        else:
            reason = "tumor-like semantics are stable, but tissue/disease semantics remain mixed"
            route_subtype = "mixed_tumor_semantics"
    else:
        reason = "semantic anchors disagree too strongly for a stable broad interpretation"
        route_subtype = "unstable_semantics"

    return {
        "route": route,
        "route_subtype": route_subtype,
        "site_family": site_label,
        "tumor_status": tumor_label,
        "disease_family": disease_label,
        "site_family_confidence": site_conf,
        "tumor_confidence": tumor_conf,
        "disease_family_confidence": disease_conf,
        "prototype_site_family": proto_site_family,
        "prototype_disease_family": proto_disease_family,
        "prototype_matches_site_family": proto_matches_site,
        "prototype_matches_disease_family": proto_matches_disease,
        "site_family_distribution": site_dist,
        "tumor_distribution": tumor_dist,
        "disease_family_distribution": disease_dist,
        "reason": reason,
    }


def parse_gene_count_file(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("Expected at least two columns: gene,count")
    out = {}
    for _, row in df.iloc[:, :2].iterrows():
        gene = str(row.iloc[0]).strip()
        if not gene or gene.lower() in {"gene", "genes"}:
            continue
        try:
            out[gene.upper()] = float(row.iloc[1])
        except Exception:
            continue
    return out


def build_expr_vector(selected_genes: list[str], counts: dict[str, float]) -> tuple[np.ndarray, int]:
    vec = np.zeros((len(selected_genes),), dtype=np.float32)
    matched = 0
    for i, g in enumerate(selected_genes):
        v = counts.get(g.upper())
        if v is None:
            continue
        vec[i] = v
        matched += 1
    return vec, matched


@lru_cache(maxsize=1)
def load_runtime() -> dict:
    cfg = json.loads(BEST_CFG.read_text())
    attn_dir = OUTPUT_ROOT / cfg["attention_run"]
    fusion_alpha = float(cfg["fusion_alpha"])
    fusion_alpha = float(os.getenv("SEMANTIC_FUSION_ALPHA_OVERRIDE", fusion_alpha))

    payload = torch.load(attn_dir / "semantic_prototype_attention.pt", map_location="cpu", weights_only=False)
    summary = payload["summary"]

    os.environ["SPTA_BASE_RUN"] = summary["base_run"]
    os.environ["SPTA_TEXT_SOURCE"] = summary["text_source"]
    os.environ["SPTA_N_PROTOTYPES"] = str(summary["n_prototypes"])
    os.environ["SPTA_TOPK_PROTOTYPES"] = str(summary.get("topk_prototypes", 0) or 0)
    module = load_module(TRAIN_SCRIPT, "semantic_unknown_explainer_train")
    art = module.load_artifacts()
    train_meta, _, _ = module.build_splits(art)
    train_expr = module.encode_expr_embeddings(art, train_meta)
    train_text = module.encode_text_embeddings(art, train_meta["sample_id"].tolist())
    proto_bank, proto_table, _ = module.fit_prototypes(train_text, train_meta, art.semantic_text_column)
    model = module.SemanticPrototypeAttention(embed_dim=proto_bank.shape[1])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "cfg": cfg,
        "fusion_alpha": fusion_alpha,
        "module": module,
        "art": art,
        "train_meta": train_meta,
        "train_expr": train_expr,
        "train_text": train_text,
        "proto_bank": proto_bank,
        "proto_table": proto_table,
        "model": model,
    }


def explain_counts(
    counts: dict[str, float],
    top_k: int = 5,
    rerank_beta: float = 0.3,
    family_consistency_gamma: float = 0.10,
) -> dict:
    rt = load_runtime()
    cfg = rt["cfg"]
    fusion_alpha = rt["fusion_alpha"]
    art = rt["art"]
    train_meta = rt["train_meta"]
    train_expr = rt["train_expr"]
    train_text = rt["train_text"]
    proto_bank = rt["proto_bank"]
    proto_table = rt["proto_table"]
    model = rt["model"]

    vec, matched = build_expr_vector(art.selected_genes, counts)
    x = np.clip((vec - art.expr_mean) / art.expr_std, -8.0, 8.0)[None, :]
    with torch.no_grad():
        direct = art.model.encode_expr(torch.tensor(x, dtype=torch.float32)).cpu().numpy().astype(np.float32)
        age_z = art.model.age_head(torch.tensor(direct, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    with torch.no_grad():
        mixed, weights = model(torch.tensor(direct, dtype=torch.float32), torch.tensor(proto_bank, dtype=torch.float32))
    direct_n = normalize_rows(direct)
    mixed_n = normalize_rows(mixed.cpu().numpy().astype(np.float32))
    fused = normalize_rows((1.0 - fusion_alpha) * direct_n + fusion_alpha * mixed_n)

    text_bank = normalize_rows(train_text)
    proto_weights = weights.cpu().numpy()[0]
    top_proto = int(proto_weights.argmax())
    top_proto_row = proto_table.iloc[top_proto]
    sims = fused @ text_bank.T
    # Prototype-conditioned rerank: reward neighbors aligned with the attention-weighted prototype mixture.
    proto_mixture = normalize_rows(proto_weights[None, :] @ normalize_rows(proto_bank))
    proto_sims = proto_mixture @ text_bank.T
    sims = sims + float(rerank_beta) * proto_sims
    # Broad-family consistency prior: keep semantic neighbors closer to the top prototype family.
    proto_site_family = organ_family(str(top_proto_row.get("feat_anatomical_site", "unknown")))
    neighbor_site_families = train_meta["feat_anatomical_site"].astype(str).map(organ_family).to_numpy()
    family_prior = np.where(neighbor_site_families == proto_site_family, 1.0, -1.0).astype(np.float32)[None, :]
    sims = sims + float(family_consistency_gamma) * family_prior
    topk = min(top_k, sims.shape[1])
    idx = np.argsort(-sims[0])[:topk]

    expr_sims = normalize_rows(direct) @ normalize_rows(train_expr).T
    expr_idx = np.argsort(-expr_sims[0])[: min(21, expr_sims.shape[1])]
    age_knn, age_knn_conf = _weighted_knn_regression(
        pd.to_numeric(train_meta.iloc[expr_idx]["feat_age"], errors="coerce").to_numpy(dtype=np.float32),
        expr_sims[0, expr_idx],
    )
    age_head_scalar = float(np.asarray(age_z).reshape(-1)[0])
    age_years_head = float(
        age_head_scalar * float(art.checkpoint.get("age_std", 1.0)) + float(art.checkpoint.get("age_mean", 0.0))
    )
    age_like_years = age_knn if age_knn is not None else age_years_head
    age_like_conf = float(age_knn_conf if age_knn is not None else 0.35)
    age_like_mode = "exact" if age_like_conf >= 0.80 else ("band" if age_like_conf >= 0.50 else "withheld")
    age_like_method = "knn" if age_knn is not None else "head"

    neighbors = []
    driver_votes: dict[str, float] = {}
    for rank, i in enumerate(idx, start=1):
        row = train_meta.iloc[int(i)]
        row_tags = _context_tags_from_row(row)
        neighbor_score = float(sims[0, i])
        for tag in row_tags:
            driver_votes[tag] = driver_votes.get(tag, 0.0) + max(neighbor_score, 0.0)
        neighbors.append(
            {
                "rank": rank,
                "sample_id": str(row["sample_id"]),
                "score": float(sims[0, i]),
                "semantic_text": str(row[art.semantic_text_column]),
                "site_anchor": str(row.get("feat_anatomical_site", "unknown")),
                "tumor_anchor": str(row.get("feat_tumor_status", "unknown")),
                "disease_anchor": str(row.get("feat_disease_label", "unknown")),
                "context_tags": row_tags,
            }
        )

    driver_list = [k for k, _ in sorted(driver_votes.items(), key=lambda kv: -kv[1])[:4]]
    structured_anchor = _compute_structured_anchor(counts)
    semantic_consensus = _build_semantic_consensus(
        {
            "site_anchor": str(top_proto_row.get("feat_anatomical_site", "unknown")),
            "tumor_anchor": str(top_proto_row.get("feat_tumor_status", "unknown")),
            "disease_anchor": str(top_proto_row.get("feat_disease_label", "unknown")),
        },
        neighbors,
    )
    semantic_consensus = _apply_anchor_demote(semantic_consensus, structured_anchor)
    age_like_state = {
        "years": float(age_like_years) if age_like_mode != "withheld" else None,
        "band": _age_band(age_like_years) if age_like_conf >= 0.50 else "not stable enough to report",
        "mode": age_like_mode,
        "confidence": age_like_conf,
        "method": age_like_method,
        "head_years": age_years_head,
        "knn_years": age_knn,
        "possible_drivers": driver_list,
    }
    semantic_explanation = _build_semantic_explanation(
        semantic_consensus=semantic_consensus,
        structured_anchor=structured_anchor,
        top_prototype={
            "site_anchor": str(top_proto_row.get("feat_anatomical_site", "unknown")),
            "tumor_anchor": str(top_proto_row.get("feat_tumor_status", "unknown")),
            "disease_anchor": str(top_proto_row.get("feat_disease_label", "unknown")),
        },
        neighbors=neighbors,
        age_like_state=age_like_state,
    )

    return {
        "semantic_mainline": cfg["name"],
        "matched_selected_genes": int(matched),
        "fusion_alpha": fusion_alpha,
        "rerank_beta": rerank_beta,
        "family_consistency_gamma": family_consistency_gamma,
        "semantic_consensus": semantic_consensus,
        "semantic_explanation": semantic_explanation,
        "structured_anchor": structured_anchor,
        "top_prototype": {
            "prototype_id": top_proto,
            "weight": float(proto_weights[top_proto]),
            "semantic_text": str(top_proto_row["prototype_semantic_text"]),
            "site_anchor": str(top_proto_row.get("feat_anatomical_site", "unknown")),
            "tumor_anchor": str(top_proto_row.get("feat_tumor_status", "unknown")),
            "disease_anchor": str(top_proto_row.get("feat_disease_label", "unknown")),
        },
        "age_like_state": age_like_state,
        "top_neighbors": neighbors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-beta", type=float, default=0.3)
    args = parser.parse_args()

    counts = parse_gene_count_file(args.input_path)
    out = explain_counts(counts, top_k=args.top_k, rerank_beta=args.rerank_beta)
    out["input_path"] = str(args.input_path)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
