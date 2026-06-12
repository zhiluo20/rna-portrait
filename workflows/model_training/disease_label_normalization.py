from __future__ import annotations

from typing import Dict


# Conservative normalization only fixes obvious surface/case/cohort-wrapper variants.
CONSERVATIVE_MAP: Dict[str, str] = {
    "lung adenocarcinoma": "Lung Adenocarcinoma",
    "Lung Adenocarcinoma Patients": "Lung Adenocarcinoma",
    "Lung Adenocarcinoma Patients without EGFR or ALK Alterations": "Lung Adenocarcinoma",
    "Never-Smoker Lung Adenocarcinoma": "Lung Adenocarcinoma",
    "Never-Smoker Lung Adenocarcinoma Patients without EGFR or ALK Alterations": "Lung Adenocarcinoma",
    "glioblastoma": "Glioblastoma",
    "uterine leiomyoma": "Uterine leiomyoma",
}


# Ontology-level normalization collapses labels that are frequently confused but
# remain coherent at a broader disease-entity level.
ONTOLOGY_MAP: Dict[str, str] = {
    **CONSERVATIVE_MAP,
    "NAFLD": "fatty_liver_disease",
    "NAFLD/NASH": "fatty_liver_disease",
    "NASH": "fatty_liver_disease",
    "COVID-19": "viral_respiratory_or_systemic_infection",
    "non-COVID-19": "viral_respiratory_or_systemic_infection",
    "Acute myeloid leukemia with mutated NPM1": "Acute myeloid leukemia, NOS",
    "Leukemias, NOS": "Leukemia",
    "Chronic lymphocytic leukemia": "Leukemia",
    "Adult T-cell leukemia/lymphoma (HTLV-1 positive) (includes all variants)": "Leukemia",
    "Mature T- and NK-Cell Lymphomas": "Leukemia",
    "Acute lymphocytic leukemia": "Leukemia",
    "Acute myeloid leukemia, NOS": "Leukemia",
    "Clear cell adenocarcinoma, NOS": "Renal epithelial carcinoma",
    "Papillary adenocarcinoma, NOS": "Renal epithelial carcinoma",
    "Renal cell carcinoma, NOS": "Renal epithelial carcinoma",
    "Renal cell carcinoma, chromophobe type": "Renal epithelial carcinoma",
    "Adenomas and Adenocarcinomas": "Renal epithelial carcinoma",
    "Lobular carcinoma, NOS": "breast_carcinoma",
    "Infiltrating duct carcinoma, NOS": "breast_carcinoma",
    "Infiltrating duct and lobular carcinoma": "breast_carcinoma",
    "Ductal and Lobular Neoplasms": "breast_carcinoma",
    "Mucinous adenocarcinoma": "breast_carcinoma",
    "primary mammary tumor": "breast_carcinoma",
    "Pheochromocytoma, NOS": "neuroendocrine_adrenal_tumor",
    "Pheochromocytoma, malignant": "neuroendocrine_adrenal_tumor",
    "Neuroblastoma, NOS": "neuroendocrine_adrenal_tumor",
}


def normalize_disease_label(label: str, strategy: str = "raw") -> str:
    s = str(label)
    if strategy == "raw":
        return s
    if strategy == "conservative":
        return CONSERVATIVE_MAP.get(s, s)
    if strategy == "ontology":
        return ONTOLOGY_MAP.get(s, s)
    raise ValueError(f"Unknown normalization strategy: {strategy}")

