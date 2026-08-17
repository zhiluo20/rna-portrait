#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from repro_paths import OUTPUT_ROOT, VALIDATION_INPUT_ROOT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION_DIR = VALIDATION_INPUT_ROOT / "external_180"
SCRIPT = Path(__file__).resolve().with_name("infer_unknown_rna_semantic_explainer.py")
DEFAULT_OUTDIR = OUTPUT_ROOT / "benchmark_semantic_unknown_explainer_expanded_runtime"


def load_module(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("semantic_unknown_validation_expanded", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    validation_dir = args.validation_dir
    manifest_path = validation_dir / "manifest.json"
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    module = load_module(SCRIPT)
    manifest = json.loads(manifest_path.read_text())
    manifest_by_file = {row["file"]: row for row in manifest}
    rows = []

    for txt in sorted(validation_dir.glob("*.txt")):
        counts = module.parse_gene_count_file(txt)
        payload = module.explain_counts(counts, top_k=3, rerank_beta=0.3)
        proto = payload["top_prototype"]
        neigh = payload["top_neighbors"]
        age_like = payload.get("age_like_state", {})
        consensus = payload.get("semantic_consensus", {})
        explanation = payload.get("semantic_explanation", {})
        mixed_ev = explanation.get("mixed_biology_evidence", {})
        state_card = explanation.get("semantic_state_card", {})
        disease_resolution = explanation.get("disease_semantic_resolution", {})
        disease_card = explanation.get("disease_semantic_card", {})
        portrait = explanation.get("semantic_portrait", {})
        evidence = list(state_card.get("evidence_highlights", []) or [])
        top_ev = evidence[0] if evidence else {}
        second_ev = evidence[1] if len(evidence) > 1 else {}
        anchor = payload.get("structured_anchor", {})
        meta = manifest_by_file.get(txt.name, {})
        expected_site_family = meta.get("expected_site_family", "unknown")
        expected_disease_family = meta.get("expected_disease_family", "unknown")
        rows.append(
            {
                "file": txt.name,
                "project": meta.get("project", "unknown"),
                "explainer_route": meta.get("explainer_route", "unknown"),
                "explainer_ok_source": int(bool(meta.get("explainer_ok", False))),
                "expected_site_family": expected_site_family,
                "expected_disease_family": expected_disease_family,
                "matched_selected_genes": payload["matched_selected_genes"],
                "prototype_site": proto["site_anchor"],
                "prototype_tumor": proto["tumor_anchor"],
                "prototype_disease": proto["disease_anchor"],
                "neighbor1_site": neigh[0]["site_anchor"] if neigh else "unknown",
                "neighbor1_tumor": neigh[0]["tumor_anchor"] if neigh else "unknown",
                "neighbor1_disease": neigh[0]["disease_anchor"] if neigh else "unknown",
                "site_agree_top1": int(bool(neigh) and proto["site_anchor"] == neigh[0]["site_anchor"]),
                "tumor_agree_top1": int(bool(neigh) and proto["tumor_anchor"] == neigh[0]["tumor_anchor"]),
                "disease_agree_top1": int(bool(neigh) and proto["disease_anchor"] == neigh[0]["disease_anchor"]),
                "age_like_mode": age_like.get("mode", "unknown"),
                "age_like_band": age_like.get("band", "unknown"),
                "semantic_route": consensus.get("route", "unknown"),
                "semantic_route_subtype": consensus.get("route_subtype", "unknown"),
                "semantic_site_family": consensus.get("site_family", "unknown"),
                "semantic_tumor_status": consensus.get("tumor_status", "unknown"),
                "semantic_disease_family": consensus.get("disease_family", "unknown"),
                "semantic_anchor_adjusted": int(bool(consensus.get("anchor_adjusted", False))),
                "semantic_explanation_mode": explanation.get("mode", "unknown"),
                "semantic_override_profile": mixed_ev.get("override_profile", "none"),
                "semantic_hematologic_subprofile": mixed_ev.get("hematologic_subprofile", "none"),
                "semantic_epithelial_subprofile": mixed_ev.get("epithelial_subprofile", "none"),
                "semantic_clean_anchor_subprofile": mixed_ev.get("clean_anchor_subprofile", "none"),
                "semantic_generic_subprofile": mixed_ev.get("generic_subprofile", "none"),
                "semantic_profile_summary": mixed_ev.get("profile_summary", ""),
                "semantic_state_family": state_card.get("state_family", "unknown"),
                "semantic_state_subprofile": state_card.get("state_subprofile", "unknown"),
                "semantic_state_evidence_kinds": "|".join([str(x.get("kind", "")) for x in (state_card.get("evidence_highlights", []) or [])]),
                "semantic_state_top_evidence_kind": (
                    (state_card.get("evidence_highlights", []) or [{}])[0].get("kind", "unknown")
                    if (state_card.get("evidence_highlights", []) or [])
                    else "unknown"
                ),
                "semantic_resolved_disease_family": disease_resolution.get("resolved_disease_family", "unknown"),
                "semantic_disease_semantic_status": disease_resolution.get("status", "unknown"),
                "semantic_disease_semantic_subprofile": disease_resolution.get("subprofile", "unknown"),
                "semantic_disease_what_to_trust": disease_card.get("what_to_trust", ""),
                "semantic_disease_biological_readout": disease_card.get("biological_readout", ""),
                "semantic_portrait_present": int(bool(portrait)),
                "semantic_portrait_headline": portrait.get("headline", ""),
                "semantic_portrait_confidence_posture": portrait.get("confidence_posture", ""),
                "semantic_portrait_biological_summary": portrait.get("biological_summary", ""),
                "semantic_portrait_what_to_trust": portrait.get("what_to_trust", ""),
                "semantic_portrait_caution": portrait.get("caution", ""),
                "semantic_portrait_recommended_reading": portrait.get("recommended_reading", ""),
                "semantic_state_evidence_json": json.dumps(evidence, ensure_ascii=False),
                "semantic_state_top_evidence_score": float(top_ev.get("score", 0.0) or 0.0),
                "semantic_state_top_evidence_strength": float(top_ev.get("strength", 0.0) or 0.0),
                "semantic_state_second_evidence_kind": str(second_ev.get("kind", "unknown")),
                "semantic_state_second_evidence_score": float(second_ev.get("score", 0.0) or 0.0),
                "semantic_state_second_evidence_strength": float(second_ev.get("strength", 0.0) or 0.0),
                "anchor_context_state": anchor.get("context_state", "unknown"),
                "anchor_tumor_status": anchor.get("tumor_status", "unknown"),
                "site_family_match_expected": int(str(consensus.get("site_family", "unknown")) == str(expected_site_family)),
                "disease_family_match_expected": int(str(consensus.get("disease_family", "unknown")) == str(expected_disease_family)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "details.csv", index=False)
    by_route = (
        df.groupby("explainer_route")[["site_family_match_expected", "disease_family_match_expected", "site_agree_top1", "tumor_agree_top1"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(
            columns={
                "site_family_match_expected": "expected_site_family_match_rate",
                "disease_family_match_expected": "expected_disease_family_match_rate",
                "site_agree_top1": "site_agree_top1_rate",
                "tumor_agree_top1": "tumor_agree_top1_rate",
            }
        )
    )
    by_route.to_csv(outdir / "by_route.csv", index=False)

    summary = {
        "n_samples": int(len(df)),
        "n_projects": int(df["project"].astype(str).nunique()) if not df.empty else 0,
        "source_route_counts": df["explainer_route"].astype(str).value_counts().to_dict() if not df.empty else {},
        "site_agree_top1_rate": float(df["site_agree_top1"].mean()) if not df.empty else None,
        "tumor_agree_top1_rate": float(df["tumor_agree_top1"].mean()) if not df.empty else None,
        "disease_agree_top1_rate": float(df["disease_agree_top1"].mean()) if not df.empty else None,
        "expected_site_family_match_rate": float(df["site_family_match_expected"].mean()) if not df.empty else None,
        "expected_disease_family_match_rate": float(df["disease_family_match_expected"].mean()) if not df.empty else None,
        "semantic_route_counts": df["semantic_route"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_route_subtype_counts": df["semantic_route_subtype"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_explanation_mode_counts": df["semantic_explanation_mode"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_override_profile_counts": df["semantic_override_profile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_hematologic_subprofile_counts": df["semantic_hematologic_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_epithelial_subprofile_counts": df["semantic_epithelial_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_clean_anchor_subprofile_counts": df["semantic_clean_anchor_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_generic_subprofile_counts": df["semantic_generic_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_profile_summary_counts": df["semantic_profile_summary"].astype(str).value_counts().head(12).to_dict() if not df.empty else {},
        "semantic_state_family_counts": df["semantic_state_family"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_state_subprofile_counts": df["semantic_state_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_state_evidence_kind_counts": (
            pd.Series(
                [
                    kind
                    for raw in df["semantic_state_evidence_kinds"].fillna("").astype(str)
                    for kind in raw.split("|")
                    if kind
                ]
            ).value_counts().to_dict()
            if not df.empty else {}
        ),
        "semantic_state_top_evidence_by_family": (
            df.groupby("semantic_state_family")["semantic_state_top_evidence_kind"]
            .agg(lambda s: s.value_counts().to_dict())
            .to_dict()
            if not df.empty else {}
        ),
        "semantic_state_top_evidence_by_subprofile": (
            df.groupby("semantic_state_subprofile")["semantic_state_top_evidence_kind"]
            .agg(lambda s: s.value_counts().to_dict())
            .to_dict()
            if not df.empty else {}
        ),
        "semantic_resolved_disease_family_counts": df["semantic_resolved_disease_family"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_disease_semantic_status_counts": df["semantic_disease_semantic_status"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_disease_semantic_subprofile_counts": df["semantic_disease_semantic_subprofile"].astype(str).value_counts().to_dict() if not df.empty else {},
        "semantic_portrait_present_rate": float(df["semantic_portrait_present"].mean()) if not df.empty else None,
        "semantic_portrait_headline_unique_count": int(df["semantic_portrait_headline"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "semantic_portrait_confidence_unique_count": int(df["semantic_portrait_confidence_posture"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "semantic_portrait_biology_unique_count": int(df["semantic_portrait_biological_summary"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "semantic_portrait_trust_unique_count": int(df["semantic_portrait_what_to_trust"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "semantic_portrait_caution_unique_count": int(df["semantic_portrait_caution"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "semantic_portrait_reading_unique_count": int(df["semantic_portrait_recommended_reading"].fillna("").astype(str).replace("", pd.NA).dropna().nunique()) if not df.empty else 0,
        "anchor_context_state_counts": df["anchor_context_state"].astype(str).value_counts().to_dict() if not df.empty else {},
        "anchor_adjusted_rate": float(df["semantic_anchor_adjusted"].mean()) if not df.empty else None,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = [
        "# Semantic Unknown Explainer Expanded Validation",
        "",
        f"- n_samples: `{summary['n_samples']}`",
        f"- n_projects: `{summary['n_projects']}`",
        f"- source_route_counts: `{summary['source_route_counts']}`",
        f"- site_agree_top1_rate: `{summary['site_agree_top1_rate']:.4f}`",
        f"- tumor_agree_top1_rate: `{summary['tumor_agree_top1_rate']:.4f}`",
        f"- disease_agree_top1_rate: `{summary['disease_agree_top1_rate']:.4f}`",
        f"- expected_site_family_match_rate: `{summary['expected_site_family_match_rate']:.4f}`",
        f"- expected_disease_family_match_rate: `{summary['expected_disease_family_match_rate']:.4f}`",
        f"- semantic_route_counts: `{summary['semantic_route_counts']}`",
        f"- semantic_route_subtype_counts: `{summary['semantic_route_subtype_counts']}`",
        f"- semantic_explanation_mode_counts: `{summary['semantic_explanation_mode_counts']}`",
        f"- semantic_override_profile_counts: `{summary['semantic_override_profile_counts']}`",
        f"- semantic_hematologic_subprofile_counts: `{summary['semantic_hematologic_subprofile_counts']}`",
        f"- semantic_epithelial_subprofile_counts: `{summary['semantic_epithelial_subprofile_counts']}`",
        f"- semantic_clean_anchor_subprofile_counts: `{summary['semantic_clean_anchor_subprofile_counts']}`",
        f"- semantic_generic_subprofile_counts: `{summary['semantic_generic_subprofile_counts']}`",
        f"- semantic_state_family_counts: `{summary['semantic_state_family_counts']}`",
        f"- semantic_state_subprofile_counts: `{summary['semantic_state_subprofile_counts']}`",
        f"- semantic_state_evidence_kind_counts: `{summary['semantic_state_evidence_kind_counts']}`",
        f"- semantic_state_top_evidence_by_family: `{summary['semantic_state_top_evidence_by_family']}`",
        f"- semantic_state_top_evidence_by_subprofile: `{summary['semantic_state_top_evidence_by_subprofile']}`",
        f"- semantic_resolved_disease_family_counts: `{summary['semantic_resolved_disease_family_counts']}`",
        f"- semantic_disease_semantic_status_counts: `{summary['semantic_disease_semantic_status_counts']}`",
        f"- semantic_disease_semantic_subprofile_counts: `{summary['semantic_disease_semantic_subprofile_counts']}`",
        f"- semantic_portrait_present_rate: `{summary['semantic_portrait_present_rate']:.4f}`",
        f"- semantic_portrait_headline_unique_count: `{summary['semantic_portrait_headline_unique_count']}`",
        f"- semantic_portrait_confidence_unique_count: `{summary['semantic_portrait_confidence_unique_count']}`",
        f"- semantic_portrait_biology_unique_count: `{summary['semantic_portrait_biology_unique_count']}`",
        f"- semantic_portrait_trust_unique_count: `{summary['semantic_portrait_trust_unique_count']}`",
        f"- semantic_portrait_caution_unique_count: `{summary['semantic_portrait_caution_unique_count']}`",
        f"- semantic_portrait_reading_unique_count: `{summary['semantic_portrait_reading_unique_count']}`",
        f"- anchor_context_state_counts: `{summary['anchor_context_state_counts']}`",
        f"- anchor_adjusted_rate: `{summary['anchor_adjusted_rate']:.4f}`",
    ]
    (outdir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
