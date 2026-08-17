#!/usr/bin/env python3
"""Build a merge-ready TCGA bulk RNA dataset with the workspace metadata schema.

This script reads raw GDC STAR gene-count exports, links each expression file
to biospecimen and clinical metadata, and writes a processed dataset whose
expression values are log1p(CPM).

Design constraints:
- Do not fabricate metadata. Free-text summaries are assembled only from fields
  present in the GDC JSON/TSV exports.
- Keep the metadata schema aligned with the existing multimodal training code.
- Be explicit that the expression matrix is derived from GDC raw counts and
  normalized here as log1p(CPM), not DESeq2-normalized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TCGA_ROOT = ROOT / "data" / "tcga_raw_source"
DEFAULT_OUTDIR = ROOT / "data" / "tcga_processed_source"
DEFAULT_GENES_PATH = ROOT / "data" / "geo_processed_source" / "genes.npy"

COUNT_SUFFIX = ".rna_seq.augmented_star_gene_counts.tsv"
UNKNOWN = "unknown"
SHARD_SIZE = 512


def clean_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.lower() in {"not reported", "unknown", "na", "n/a", "--", "null", "none"}:
        return ""
    return s


def slug_text(value: object, default: str = UNKNOWN) -> str:
    s = clean_value(value)
    if not s:
        return default
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or default


def first_nonempty(*values: object) -> str:
    for value in values:
        s = clean_value(value)
        if s:
            return s
    return ""


def parse_age_years(clinical: dict) -> Tuple[float, str]:
    demo = clinical.get("demographic") or {}
    age_raw = ""
    for key in ("age_at_index", "days_to_birth", "year_of_birth"):
        if key in demo and demo.get(key) not in (None, ""):
            age_raw = f"{key}={demo.get(key)}"
            break

    age_at_index = demo.get("age_at_index")
    if age_at_index not in (None, ""):
        try:
            return float(age_at_index), f"{age_at_index} years"
        except Exception:
            pass

    days_to_birth = demo.get("days_to_birth")
    if days_to_birth not in (None, ""):
        try:
            years = abs(float(days_to_birth)) / 365.25
            return round(years, 1), f"{days_to_birth} days_to_birth"
        except Exception:
            pass

    return math.nan, age_raw


def normalize_sex(clinical: dict) -> str:
    demo = clinical.get("demographic") or {}
    raw = first_nonempty(demo.get("gender"), demo.get("sex_at_birth"))
    s = raw.lower()
    if s in {"male", "m"}:
        return "male"
    if s in {"female", "f"}:
        return "female"
    return UNKNOWN


def normalize_ethnicity(clinical: dict) -> str:
    demo = clinical.get("demographic") or {}
    ethnicity = clean_value(demo.get("ethnicity")).lower()
    race = clean_value(demo.get("race")).lower()
    if ethnicity == "hispanic or latino":
        return "hispanic"
    mapping = {
        "white": "white",
        "black or african american": "black",
        "asian": "asian",
        "american indian or alaska native": "other",
        "native hawaiian or other pacific islander": "other",
    }
    if race in mapping:
        return mapping[race]
    if ethnicity:
        return slug_text(ethnicity)
    return UNKNOWN


def normalize_population(_: dict) -> str:
    return "general"


def choose_diagnosis(clinical: dict) -> dict:
    diagnoses = clinical.get("diagnoses") or []
    if diagnoses:
        return diagnoses[0]
    return {}


def normalize_anatomical_site(sample: dict, clinical: dict) -> str:
    diag = choose_diagnosis(clinical)
    joined = " ".join(
        x.lower()
        for x in [
            first_nonempty(
                diag.get("tissue_or_organ_of_origin"),
                diag.get("site_of_resection_or_biopsy"),
                clinical.get("primary_site"),
                sample.get("specimen_type"),
                sample.get("sample_type"),
            )
        ]
        if x
    )
    rules = [
        ("bone marrow", "bone_marrow"),
        ("blood", "blood"),
        ("marrow", "bone_marrow"),
        ("kidney", "kidney"),
        ("liver", "liver"),
        ("pancre", "pancreas"),
        ("stomach", "stomach"),
        ("colon", "colon"),
        ("rect", "rectum"),
        ("prostate", "prostate"),
        ("bladder", "bladder"),
        ("breast", "breast"),
        ("brain", "brain"),
        ("heart", "heart"),
        ("muscle", "muscle"),
        ("skin", "skin"),
        ("lymph", "lymph_node"),
        ("pleura", "pleura"),
        ("lung", "lung"),
        ("hematopoietic", "blood"),
        ("reticuloendothelial", "blood"),
        ("head and neck", "head_and_neck"),
        ("head", "head_and_neck"),
    ]
    for needle, label in rules:
        if needle in joined:
            return label
    return "other" if joined else UNKNOWN


def normalize_biospecimen_type(sample: dict, anatomical_site: str) -> str:
    joined = " ".join(
        x.lower()
        for x in [
            first_nonempty(sample.get("sample_type"), sample.get("specimen_type"), sample.get("tissue_type"))
        ]
        if x
    )
    if "peripheral blood" in joined or "blood derived" in joined or anatomical_site == "blood":
        return "whole_blood"
    if "bone marrow" in joined:
        return "other"
    if "plasma" in joined:
        return "plasma"
    if "serum" in joined:
        return "serum"
    if "normal" in joined:
        return "normal_tissue"
    if "tumor" in joined or "cancer" in joined:
        return "tumor_tissue"
    if "solid tissue" in joined:
        return "tumor_tissue" if sample.get("tissue_type") == "Tumor" else "normal_tissue"
    return "other" if joined else UNKNOWN


def normalize_tumor_status(sample: dict, source_label: str) -> str:
    tissue_type = clean_value(sample.get("tissue_type")).lower()
    sample_type = clean_value(sample.get("sample_type")).lower()
    descriptor = clean_value(sample.get("tumor_descriptor")).lower()
    if "metastatic" in descriptor or "metastatic" in sample_type:
        return "metastatic"
    if tissue_type == "tumor" or "primary tumor" in sample_type or "recurrent tumor" in sample_type:
        return "tumor"
    if "adjacent" in sample_type:
        return "adjacent_normal"
    if tissue_type == "normal" or "normal" in sample_type or source_label == "normal":
        return "non_tumor"
    if descriptor == "not applicable":
        return "not_applicable"
    return UNKNOWN


def normalize_tissue_context(sample: dict, tumor_status: str, anatomical_site: str) -> str:
    sample_type = clean_value(sample.get("sample_type")).lower()
    if anatomical_site == "blood":
        return "blood"
    if tumor_status in {"tumor", "metastatic"}:
        return "tumor_core"
    if tumor_status == "adjacent_normal" or "adjacent" in sample_type:
        return "non_tumor_adjacent"
    if tumor_status in {"non_tumor", "not_applicable"}:
        return "healthy_tissue"
    return UNKNOWN


def normalize_disease_label(clinical: dict, tumor_status: str) -> str:
    diag = choose_diagnosis(clinical)
    diagnosis = first_nonempty(
        diag.get("primary_diagnosis"),
        clinical.get("disease_type"),
    )
    if diagnosis:
        return diagnosis
    if tumor_status in {"non_tumor", "adjacent_normal", "not_applicable"}:
        return "healthy"
    return UNKNOWN


def normalize_disease_severity(clinical: dict) -> str:
    diag = choose_diagnosis(clinical)
    for key in (
        "ajcc_pathologic_stage",
        "ajcc_clinical_stage",
        "figo_stage",
        "iss_stage",
        "ann_arbor_clinical_stage",
        "eln_risk_classification",
        "tumor_grade",
        "last_known_disease_status",
    ):
        value = clean_value(diag.get(key))
        if value:
            return value
    return UNKNOWN


def normalize_sample_role(sample: dict, tumor_status: str) -> str:
    sample_type = clean_value(sample.get("sample_type")).lower()
    if "normal" in sample_type or tumor_status in {"non_tumor", "adjacent_normal", "not_applicable"}:
        return "control"
    if tumor_status in {"tumor", "metastatic"}:
        return "case"
    return UNKNOWN


def build_metadata_text(
    sample_id: str,
    project_id: str,
    age: float,
    sex: str,
    sample: dict,
    clinical: dict,
    disease_label: str,
) -> str:
    parts: List[str] = ["RNA-seq sample"]
    if sex != UNKNOWN:
        parts.append(f"from a {sex} individual")
    if not math.isnan(age):
        parts.append(f"age {int(round(age))}")

    sample_type = clean_value(sample.get("sample_type"))
    if sample_type:
        parts.append(f"sample type {sample_type}")

    specimen_type = clean_value(sample.get("specimen_type"))
    if specimen_type:
        parts.append(f"specimen type {specimen_type}")

    tissue_type = clean_value(sample.get("tissue_type"))
    if tissue_type:
        parts.append(f"tissue type {tissue_type}")

    primary_site = clean_value(clinical.get("primary_site"))
    if primary_site:
        parts.append(f"primary site {primary_site}")

    diag = choose_diagnosis(clinical)
    organ = clean_value(diag.get("tissue_or_organ_of_origin"))
    if organ:
        parts.append(f"organ of origin {organ}")

    if disease_label != UNKNOWN:
        parts.append(f"diagnosis {disease_label}")

    parts.append(f"project {project_id}")
    parts.append(f"sample id {sample_id}")
    return ", ".join(parts) + "."


def build_caption(row: pd.Series) -> str:
    free = clean_value(row.get("metadata_text"))
    parts = []
    if pd.notna(row.get("feat_age")):
        parts.append(f"age {int(round(float(row['feat_age'])))}")
    parts.append(f"sex {row.get('feat_sex', UNKNOWN)}")
    parts.append(f"biospecimen {row.get('feat_biospecimen_type', UNKNOWN)}")
    parts.append(f"anatomical site {row.get('feat_anatomical_site', UNKNOWN)}")
    parts.append(f"tumor status {row.get('feat_tumor_status', UNKNOWN)}")
    parts.append(f"tissue context {row.get('feat_tissue_context', UNKNOWN)}")
    parts.append(f"disease {row.get('feat_disease_label', UNKNOWN)}")
    parts.append(f"disease severity {row.get('feat_disease_severity', UNKNOWN)}")
    parts.append(f"sample role {row.get('feat_sample_role', UNKNOWN)}")
    parts.append(f"gse {row.get('gse', UNKNOWN)}")
    structured = "Structured metadata: " + "; ".join(parts) + "."
    return f"{free} {structured}".strip() if free else structured


@dataclass
class SampleLink:
    sample_id: str
    sample_submitter_id: str
    case_id: str
    case_submitter_id: str
    project_id: str
    sample_type: str
    tissue_type: str
    specimen_type: str
    tumor_descriptor: str
    preservation_method: str


@dataclass
class SampleRecord:
    sample_id: str
    source_group: str
    source_path: str
    file_id: str
    file_name: str
    case_id: str
    case_submitter_id: str
    project_id: str
    tcga_sample_submitter_id: str
    tcga_sample_type: str
    tcga_tissue_type: str
    tcga_specimen_type: str
    tcga_tumor_descriptor: str
    tcga_preservation_method: str
    metadata_text: str
    feat_age: float
    feat_age_raw: str
    feat_sex: str
    feat_biospecimen_type: str
    feat_anatomical_site: str
    feat_tumor_status: str
    feat_tissue_context: str
    feat_disease_label: str
    feat_disease_severity: str
    feat_sample_role: str
    feat_ethnicity: str
    feat_population: str
    gse: str


def read_json(path: Path) -> list:
    with path.open() as fh:
        return json.load(fh)


def read_first_json(base: Path, pattern: str) -> Optional[Path]:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


def read_sample_sheet(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, dict] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            file_name = clean_value(row.get("File Name"))
            if file_name:
                out[file_name] = row
    return out


def build_biospecimen_maps(rows: Sequence[dict]) -> Tuple[Dict[str, SampleLink], Dict[str, SampleLink], Dict[str, SampleLink]]:
    by_sample_id: Dict[str, SampleLink] = {}
    by_aliquot_id: Dict[str, SampleLink] = {}
    by_aliquot_submitter: Dict[str, SampleLink] = {}

    for case in rows:
        case_id = clean_value(case.get("case_id"))
        case_submitter_id = clean_value(case.get("submitter_id"))
        project_id = clean_value((case.get("project") or {}).get("project_id"))
        for sample in case.get("samples") or []:
            link = SampleLink(
                sample_id=clean_value(sample.get("sample_id")),
                sample_submitter_id=clean_value(sample.get("submitter_id")),
                case_id=case_id,
                case_submitter_id=case_submitter_id,
                project_id=project_id,
                sample_type=clean_value(sample.get("sample_type")),
                tissue_type=clean_value(sample.get("tissue_type")),
                specimen_type=clean_value(sample.get("specimen_type")),
                tumor_descriptor=clean_value(sample.get("tumor_descriptor")),
                preservation_method=clean_value(sample.get("preservation_method")),
            )
            if link.sample_id:
                by_sample_id[link.sample_id] = link

            for portion in sample.get("portions") or []:
                for analyte in portion.get("analytes") or []:
                    for aliquot in analyte.get("aliquots") or []:
                        aliquot_id = clean_value(aliquot.get("aliquot_id"))
                        aliquot_submitter = clean_value(aliquot.get("submitter_id"))
                        if aliquot_id:
                            by_aliquot_id[aliquot_id] = link
                        if aliquot_submitter:
                            by_aliquot_submitter[aliquot_submitter] = link

    return by_sample_id, by_aliquot_id, by_aliquot_submitter


def build_clinical_map(rows: Sequence[dict]) -> Dict[str, dict]:
    return {clean_value(row.get("case_id")): row for row in rows if clean_value(row.get("case_id"))}


def build_metadata_map(rows: Sequence[dict]) -> Dict[str, dict]:
    return {clean_value(row.get("file_name")): row for row in rows if clean_value(row.get("file_name"))}


def resolve_sample_link(
    file_name: str,
    source_group: str,
    sample_sheet: Dict[str, dict],
    metadata_map: Dict[str, dict],
    by_sample_id: Dict[str, SampleLink],
    by_aliquot_id: Dict[str, SampleLink],
    by_aliquot_submitter: Dict[str, SampleLink],
) -> Tuple[Optional[SampleLink], dict]:
    sheet_row = sample_sheet.get(file_name)
    if sheet_row:
        sample_id = clean_value(sheet_row.get("Sample ID")).split(",")[0].strip()
        if sample_id:
            for link in by_sample_id.values():
                if link.sample_submitter_id == sample_id or link.sample_id == sample_id:
                    return link, metadata_map.get(file_name, {})

    meta_row = metadata_map.get(file_name, {})
    for entity in meta_row.get("associated_entities") or []:
        entity_id = clean_value(entity.get("entity_id"))
        entity_submitter_id = clean_value(entity.get("entity_submitter_id"))
        if entity_id and entity_id in by_aliquot_id:
            return by_aliquot_id[entity_id], meta_row
        if entity_submitter_id and entity_submitter_id in by_aliquot_submitter:
            return by_aliquot_submitter[entity_submitter_id], meta_row

    return None, meta_row


def parse_count_file(path: Path) -> Dict[str, float]:
    gene_counts: Dict[str, float] = {}
    with path.open() as fh:
        reader = csv.DictReader((line for line in fh if not line.startswith("#")), delimiter="\t")
        for row in reader:
            gene_id = clean_value(row.get("gene_id"))
            gene_name = clean_value(row.get("gene_name")).upper()
            if not gene_id.startswith("ENSG") or not gene_name:
                continue
            value = row.get("unstranded")
            try:
                count = float(value)
            except Exception:
                continue
            gene_counts[gene_name] = gene_counts.get(gene_name, 0.0) + count
    return gene_counts


def iter_tcga_sources(root: Path) -> List[Tuple[str, Path]]:
    sources: List[Tuple[str, Path]] = []
    normal_dir = root / "normal"
    if normal_dir.exists():
        sources.append(("normal", normal_dir))

    tumor_dir = root / "tumor"
    if tumor_dir.exists():
        for child in sorted(p for p in tumor_dir.iterdir() if p.is_dir()):
            sources.append((f"tumor/{child.name}", child))
    return sources


def count_matrix_root(source_path: Path) -> Path:
    if (source_path / "count_matrix").exists():
        return source_path / "count_matrix"
    return source_path


def find_sample_sheet(source_path: Path) -> Optional[Path]:
    matches = sorted(source_path.glob("meta/gdc_sample_sheet*.tsv"))
    return matches[0] if matches else None


def ensure_unique_sample_ids(records: List[SampleRecord]) -> None:
    seen = Counter(r.sample_id for r in records)
    used: Counter[str] = Counter()
    for record in records:
        if seen[record.sample_id] > 1:
            used[record.sample_id] += 1
            record.sample_id = f"{record.sample_id}__{record.file_id[:8]}_{used[record.sample_id]}"


def build_manifest(tcga_root: Path) -> pd.DataFrame:
    sample_records: List[SampleRecord] = []

    for source_group, source_path in iter_tcga_sources(tcga_root):
        meta_dir = source_path / "meta"
        metadata_path = read_first_json(meta_dir, "metadata.cart*.json")
        clinical_path = read_first_json(meta_dir, "clinical.cart*.json")
        biospecimen_path = read_first_json(meta_dir, "biospecimen.cart*.json")
        if not metadata_path or not clinical_path or not biospecimen_path:
            continue

        metadata_rows = read_json(metadata_path)
        clinical_rows = read_json(clinical_path)
        biospecimen_rows = read_json(biospecimen_path)
        sample_sheet = read_sample_sheet(find_sample_sheet(source_path))

        metadata_map = build_metadata_map(metadata_rows)
        clinical_map = build_clinical_map(clinical_rows)
        by_sample_id, by_aliquot_id, by_aliquot_submitter = build_biospecimen_maps(biospecimen_rows)

        for count_path in sorted(count_matrix_root(source_path).rglob(f"*{COUNT_SUFFIX}")):
            file_name = count_path.name
            link, meta_row = resolve_sample_link(
                file_name=file_name,
                source_group=source_group,
                sample_sheet=sample_sheet,
                metadata_map=metadata_map,
                by_sample_id=by_sample_id,
                by_aliquot_id=by_aliquot_id,
                by_aliquot_submitter=by_aliquot_submitter,
            )
            if link is None:
                continue

            clinical = clinical_map.get(link.case_id, {})
            age, age_raw = parse_age_years(clinical)
            sex = normalize_sex(clinical)
            anatomical_site = normalize_anatomical_site(link.__dict__, clinical)
            biospecimen_type = normalize_biospecimen_type(link.__dict__, anatomical_site)
            tumor_status = normalize_tumor_status(link.__dict__, source_group.split("/")[0])
            tissue_context = normalize_tissue_context(link.__dict__, tumor_status, anatomical_site)
            disease_label = normalize_disease_label(clinical, tumor_status)
            disease_severity = normalize_disease_severity(clinical)
            sample_role = normalize_sample_role(link.__dict__, tumor_status)
            project_id = link.project_id or clean_value((clinical.get("project") or {}).get("project_id")) or "TCGA"
            gse = f"TCGA::{project_id}"
            sample_id = link.sample_submitter_id or link.sample_id or count_path.stem

            metadata_text = build_metadata_text(
                sample_id=sample_id,
                project_id=project_id,
                age=age,
                sex=sex,
                sample=link.__dict__,
                clinical=clinical,
                disease_label=disease_label,
            )

            record = SampleRecord(
                sample_id=sample_id,
                source_group=source_group,
                source_path=str(count_path.relative_to(tcga_root)),
                file_id=clean_value(meta_row.get("file_id")) or count_path.parent.name,
                file_name=file_name,
                case_id=link.case_id,
                case_submitter_id=link.case_submitter_id,
                project_id=project_id,
                tcga_sample_submitter_id=link.sample_submitter_id,
                tcga_sample_type=link.sample_type,
                tcga_tissue_type=link.tissue_type,
                tcga_specimen_type=link.specimen_type,
                tcga_tumor_descriptor=link.tumor_descriptor,
                tcga_preservation_method=link.preservation_method,
                metadata_text=metadata_text,
                feat_age=age,
                feat_age_raw=age_raw,
                feat_sex=sex,
                feat_biospecimen_type=biospecimen_type,
                feat_anatomical_site=anatomical_site,
                feat_tumor_status=tumor_status,
                feat_tissue_context=tissue_context,
                feat_disease_label=disease_label,
                feat_disease_severity=disease_severity,
                feat_sample_role=sample_role,
                feat_ethnicity=normalize_ethnicity(clinical),
                feat_population=normalize_population(clinical),
                gse=gse,
            )
            sample_records.append(record)

    ensure_unique_sample_ids(sample_records)

    meta_df = pd.DataFrame([r.__dict__ for r in sample_records])
    if meta_df.empty:
        raise RuntimeError("No TCGA samples were linked successfully.")

    meta_df["caption_text"] = meta_df.apply(build_caption, axis=1)
    meta_df = meta_df.sort_values(["gse", "sample_id"]).reset_index(drop=True)
    return meta_df


def counts_to_log1p_cpm(expr_counts: pd.DataFrame) -> pd.DataFrame:
    libsize = expr_counts.sum(axis=0).replace(0, np.nan)
    cpm = expr_counts.divide(libsize, axis=1) * 1e6
    cpm = cpm.fillna(0.0)
    expr_log = np.log1p(cpm).astype(np.float32)
    return pd.DataFrame(expr_log, index=expr_counts.index, columns=expr_counts.columns)


def load_target_genes(genes_path: Path) -> np.ndarray:
    if genes_path.exists():
        genes = np.load(genes_path, allow_pickle=True)
        return genes.astype(str)
    return np.array([], dtype=object)


def safe_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_expression_chunk(
    meta_chunk: pd.DataFrame,
    target_genes: np.ndarray,
    tcga_root: Path,
) -> pd.DataFrame:
    gene_to_idx = {gene: idx for idx, gene in enumerate(target_genes)}
    matrix = np.zeros((len(target_genes), len(meta_chunk)), dtype=np.float32)
    sample_ids = meta_chunk["sample_id"].tolist()
    for col_idx, row in enumerate(meta_chunk.itertuples(index=False)):
        counts = parse_count_file(tcga_root / row.source_path)
        for gene, value in counts.items():
            idx = gene_to_idx.get(gene)
            if idx is not None:
                matrix[idx, col_idx] = value
    expr_counts = pd.DataFrame(matrix, index=target_genes, columns=sample_ids, dtype=np.float32)
    expr_counts.index.name = "gene_symbol"
    return expr_counts


def save_project_shards(
    meta_df: pd.DataFrame,
    outdir: Path,
    target_genes: np.ndarray,
    tcga_root: Path,
    shard_size: int = SHARD_SIZE,
) -> Dict[str, int]:
    projects_root = outdir / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    shard_counts: Dict[str, int] = {}

    for gse, project_meta in meta_df.groupby("gse", sort=True):
        project_dir = projects_root / safe_dirname(gse)
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        project_meta = project_meta.reset_index(drop=True)
        n_shards = math.ceil(len(project_meta) / shard_size)
        shard_counts[gse] = n_shards

        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, len(project_meta))
            shard_meta = project_meta.iloc[start:end].copy().reset_index(drop=True)
            shard_name = f"shard_{shard_idx:04d}"
            shard_dir = project_dir / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)

            expr_counts = build_expression_chunk(shard_meta, target_genes, tcga_root)
            expr_log = counts_to_log1p_cpm(expr_counts)

            shard_meta["project_shard"] = shard_name
            shard_meta.to_parquet(shard_dir / "meta.parquet", index=False)
            np.save(shard_dir / "sample_ids.npy", shard_meta["sample_id"].to_numpy(), allow_pickle=True)
            np.save(shard_dir / "genes.npy", target_genes, allow_pickle=True)
            expr_counts.to_parquet(shard_dir / "expr_counts.parquet")
            expr_log.to_parquet(shard_dir / "expr_log1p_cpm.parquet")

    return shard_counts


def save_outputs(
    meta_df: pd.DataFrame,
    outdir: Path,
    genes_path: Path,
    tcga_root: Path,
    shard_size: int = SHARD_SIZE,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    sample_ids = meta_df["sample_id"].to_numpy()
    genes = load_target_genes(genes_path)
    if genes.size == 0:
        raise RuntimeError(f"Missing or empty target gene list: {genes_path}")

    np.save(outdir / "sample_ids.npy", sample_ids, allow_pickle=True)
    np.save(outdir / "genes.npy", genes, allow_pickle=True)
    meta_df.to_parquet(outdir / "meta.parquet", index=False)
    meta_df.to_csv(outdir / "meta.csv", index=False)
    (outdir / "sample_ids.txt").write_text("\n".join(sample_ids.astype(str)) + "\n")
    (outdir / "genes.txt").write_text("\n".join(genes.astype(str)) + "\n")

    shard_counts = save_project_shards(
        meta_df,
        outdir,
        genes,
        tcga_root,
        shard_size=shard_size,
    )

    summary = {
        "n_samples": int(len(meta_df)),
        "n_genes": int(len(genes)),
        "n_projects": int(meta_df["gse"].nunique()),
        "projects": meta_df["gse"].value_counts().sort_index().to_dict(),
        "project_shards": shard_counts,
        "tumor_status": meta_df["feat_tumor_status"].value_counts(dropna=False).to_dict(),
        "biospecimen_type": meta_df["feat_biospecimen_type"].value_counts(dropna=False).to_dict(),
        "anatomical_site_top20": meta_df["feat_anatomical_site"].value_counts().head(20).to_dict(),
        "expression_layout": {
            "type": "project_sharded_dense_parquet",
            "root": "projects/<gse>/shard_xxxx/",
            "shard_size": shard_size,
        },
        "expression_matrix": "Each shard stores expr_counts.parquet and expr_log1p_cpm.parquet using the supplied gene set; expr_log1p_cpm is log1p(CPM) from GDC unstranded counts and is not DESeq2-normalized.",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcga-root", type=Path, default=TCGA_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--genes",
        type=Path,
        default=DEFAULT_GENES_PATH,
        help="genes.npy defining the ordered feature space shared with the GEO matrix.",
    )
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    args = parser.parse_args()

    tcga_root = args.tcga_root.resolve()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    meta_df = build_manifest(tcga_root)
    save_outputs(
        meta_df,
        args.outdir.resolve(),
        args.genes.resolve(),
        tcga_root,
        shard_size=args.shard_size,
    )

    print(f"[OK] saved processed TCGA dataset to {args.outdir}")
    print(
        f"      samples={len(meta_df)} "
        f"genes={len(load_target_genes(args.genes.resolve()))} "
        f"projects={meta_df['gse'].nunique()}"
    )


if __name__ == "__main__":
    main()
