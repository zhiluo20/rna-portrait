from __future__ import annotations

import re
from collections import Counter


ORGAN_FAMILY_MAP = {
    "hematologic": {"blood", "bone_marrow", "lymph_node"},
    "thoracic": {"lung", "pleura", "head_and_neck", "larynx"},
    "digestive": {"liver", "colon", "rectum", "stomach", "pancreas"},
    "genitourinary": {"kidney", "bladder", "prostate"},
    "cns": {"brain"},
    "cutaneous": {"skin"},
    "breast": {"breast"},
    "other": {"other", "unknown", "muscle", "heart"},
}


def organ_family(site: str) -> str:
    value = str(site or "unknown").strip().lower()
    for family, members in ORGAN_FAMILY_MAP.items():
        if value in members:
            return family
    return "other"


def _contains_phrase(label: str, phrases: list[str]) -> bool:
    return any(phrase in label for phrase in phrases)


def _contains_token(label: str, token: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", label) is not None


def disease_family(disease_label: str, anatomical_site: str | None = None, tumor_status: str | None = None) -> str:
    label = str(disease_label or "unknown").strip().lower()
    site = organ_family(anatomical_site or "unknown")
    tumor = str(tumor_status or "unknown").strip().lower()

    if label in {"unknown", "nan", "none", "na", ""}:
        if tumor in {"non_tumor", "adjacent_normal", "not_applicable"}:
            return "healthy_control"
        return f"{site}_unknown"

    if _contains_phrase(label, ["healthy", "control", "normal", "adjacent normal"]):
        return "healthy_control"
    if _contains_phrase(label, ["covid", "sars", "influenza", "sepsis", "infection", "hbv", "hcv", "viral", "hepatitis"]):
        return "infectious"
    if _contains_phrase(label, ["aml", "leukemia", "lymphoma", "myeloma", "myeloproliferative", "hematologic", "b-cell", "t-cell"]):
        return "hematologic_malignancy"
    if _contains_phrase(label, ["nafld", "nash", "fatty liver", "steato", "fibrosis"]) and site == "digestive":
        return "liver_metabolic"
    if _contains_phrase(label, ["nsclc", "lung adenocarcinoma", "lung squamous", "mesothelioma", "thymoma"]):
        return "thoracic_tumor"
    if _contains_phrase(label, ["breast", "mammary", "tnbc", "duct carcinoma", "lobular carcinoma"]):
        return "breast_tumor"
    if _contains_phrase(label, ["glioblastoma", "glioma", "astrocytoma", "brain tumor", "oligodendroglioma"]):
        return "cns_tumor"
    if _contains_phrase(label, ["melanoma", "skin tumor", "cutaneous", "basal cell", "squamous cell skin"]):
        return "cutaneous_tumor"
    if _contains_phrase(label, ["kidney", "renal", "urothelial", "bladder", "prostate", "chromophobe"]):
        return "genitourinary_tumor"
    if _contains_phrase(label, ["colon", "colorectal", "rectal", "stomach", "gastric", "pancreatic", "liver cancer", "hepatocellular", "cholangio", "adenocarcinoma"]) and site == "digestive":
        return "digestive_tumor"
    if (
        _contains_phrase(label, ["crohn", "ulcerative", "ibd", "sle", "sjogren", "pcos", "endometriosis", "nfib", "autoimmune", "inflammatory", "rheumatoid arthritis"])
        or _contains_token(label, "ra")
    ):
        return "autoimmune_inflammatory"
    if _contains_phrase(label, ["cardio", "heart failure", "athero", "myocard", "coronary"]):
        return "cardiovascular"
    if _contains_phrase(label, ["muscular", "myopathy", "myotube", "muscle", "dystrophy"]):
        return "musculoskeletal"
    if tumor in {"tumor", "metastatic"}:
        return f"{site}_solid_tumor"
    if tumor in {"non_tumor", "adjacent_normal", "not_applicable"}:
        return f"{site}_non_tumor"
    return f"{site}_other"


def majority_label(values: list[str], default: str = "unknown") -> str:
    clean = [str(v) for v in values if str(v) not in {"nan", "None", ""}]
    if not clean:
        return default
    return Counter(clean).most_common(1)[0][0]
