from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .io import expression_vector, read_gene_expression_table
from .models import BulkRNALanguageAligner, PortraitAttention
from .taxonomy import disease_family, organ_family


def _default_model_root() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


def _clean_value(value: object, default: str = "unknown") -> str:
    text = str(value)
    if text.lower() in {"", "nan", "none"}:
        return default
    return text


@dataclass(frozen=True)
class ModelPaths:
    alignment_checkpoint: Path
    portrait_checkpoint: Path
    prototype_table: Path
    runtime_config: Path

    @classmethod
    def from_model_root(cls, model_root: str | Path | None = None) -> "ModelPaths":
        root = Path(model_root).resolve() if model_root is not None else _default_model_root().resolve()
        return cls(
            alignment_checkpoint=root / "rna_language_alignment" / "rna_language_alignment.pt",
            portrait_checkpoint=root / "portrait_attention" / "portrait_attention.pt",
            prototype_table=root / "portrait_attention" / "portrait_prototypes.csv",
            runtime_config=root / "portrait_attention" / "portrait_runtime_config.json",
        )


class RNAPortraitModel:
    """Load trained RNA-language models and describe a bulk expression profile."""

    def __init__(
        self,
        alignment_model: BulkRNALanguageAligner,
        portrait_model: PortraitAttention,
        prototype_bank: np.ndarray,
        prototype_table: pd.DataFrame,
        selected_genes: list[str],
        expr_mean: np.ndarray,
        expr_std: np.ndarray,
        age_mean: float,
        age_std: float,
        fusion_alpha: float,
        device: torch.device,
    ):
        self.alignment_model = alignment_model
        self.portrait_model = portrait_model
        self.prototype_bank = prototype_bank.astype(np.float32)
        self.prototype_table = prototype_table.reset_index(drop=True)
        self.selected_genes = [str(g) for g in selected_genes]
        self.expr_mean = expr_mean.astype(np.float32)
        self.expr_std = np.where(expr_std.astype(np.float32) < 1e-6, 1.0, expr_std.astype(np.float32))
        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if abs(float(age_std)) > 1e-6 else 1.0
        self.fusion_alpha = float(fusion_alpha)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_root: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "RNAPortraitModel":
        paths = ModelPaths.from_model_root(model_root)
        device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        alignment_payload = torch.load(paths.alignment_checkpoint, map_location="cpu", weights_only=False)
        n_genes = len(alignment_payload["selected_genes"])
        n_text_features = int(alignment_payload["config"]["n_text_features"])
        n_sources = len(alignment_payload["source_map"])
        alignment_model = BulkRNALanguageAligner(n_genes=n_genes, n_text_features=n_text_features, n_sources=n_sources)
        alignment_model.load_state_dict(alignment_payload["state_dict"])
        alignment_model.to(device_obj).eval()

        portrait_payload = torch.load(paths.portrait_checkpoint, map_location="cpu", weights_only=False)
        summary = portrait_payload.get("summary", {})
        state = portrait_payload["state_dict"]
        hidden_dim = int(state["post.0.weight"].shape[0])
        n_heads = int(summary.get("n_heads", 8))
        topk_prototypes = int(summary.get("topk_prototypes", 0) or 0)
        prototype_bank = np.asarray(portrait_payload["prototype_bank"], dtype=np.float32)
        portrait_model = PortraitAttention(
            embed_dim=prototype_bank.shape[1],
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            topk_prototypes=topk_prototypes,
        )
        portrait_model.load_state_dict(state)
        portrait_model.to(device_obj).eval()

        prototype_table = pd.read_csv(paths.prototype_table)
        if paths.runtime_config.exists():
            runtime_config = json.loads(paths.runtime_config.read_text(encoding="utf-8"))
            fusion_alpha = runtime_config.get("fusion_alpha", 0.4)
        else:
            fusion_alpha = 0.4

        return cls(
            alignment_model=alignment_model,
            portrait_model=portrait_model,
            prototype_bank=prototype_bank,
            prototype_table=prototype_table,
            selected_genes=list(alignment_payload["selected_genes"]),
            expr_mean=np.asarray(alignment_payload["expr_mean"], dtype=np.float32),
            expr_std=np.asarray(alignment_payload["expr_std"], dtype=np.float32),
            age_mean=float(alignment_payload.get("age_mean", 0.0)),
            age_std=float(alignment_payload.get("age_std", 1.0)),
            fusion_alpha=float(fusion_alpha),
            device=device_obj,
        )

    def encode_expression(
        self,
        expression_values: Mapping[str, float],
        log1p: bool = False,
    ) -> dict[str, object]:
        vector, matched = expression_vector(self.selected_genes, expression_values, log1p=log1p)
        x = np.clip((vector - self.expr_mean) / self.expr_std, -8.0, 8.0)[None, :]
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        proto_tensor = torch.tensor(self.prototype_bank, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            direct = self.alignment_model.encode_expr(x_tensor)
            age_z = self.alignment_model.age_head(direct).reshape(-1)
            tumor_logit = self.alignment_model.tumor_head(direct).reshape(-1)
            portrait, weights = self.portrait_model(direct, proto_tensor)

        direct_np = direct.detach().cpu().numpy().astype(np.float32)
        portrait_np = portrait.detach().cpu().numpy().astype(np.float32)
        fused = _normalize_rows((1.0 - self.fusion_alpha) * direct_np + self.fusion_alpha * portrait_np)
        return {
            "matched_genes": int(matched),
            "matched_gene_fraction": float(matched / max(len(self.selected_genes), 1)),
            "direct_embedding": direct_np[0],
            "portrait_embedding": portrait_np[0],
            "fused_embedding": fused[0],
            "prototype_weights": weights.detach().cpu().numpy().astype(np.float32)[0],
            "age_like_years": float(age_z.detach().cpu().numpy()[0] * self.age_std + self.age_mean),
            "tumor_probability": float(torch.sigmoid(tumor_logit).detach().cpu().numpy()[0]),
        }

    def describe(
        self,
        expression_values: Mapping[str, float],
        top_k: int = 5,
        log1p: bool = False,
    ) -> dict[str, object]:
        encoded = self.encode_expression(expression_values, log1p=log1p)
        weights = np.asarray(encoded["prototype_weights"], dtype=np.float32)
        top_indices = np.argsort(-weights)[: int(top_k)]

        components = []
        for rank, index in enumerate(top_indices, start=1):
            row = self.prototype_table.iloc[int(index)]
            site = _clean_value(row.get("feat_anatomical_site", "unknown"))
            tumor_status = _clean_value(row.get("feat_tumor_status", "unknown"))
            disease = _clean_value(row.get("feat_disease_label", "unknown"))
            components.append(
                {
                    "rank": rank,
                    "prototype_id": int(row.get("prototype_id", index)),
                    "weight": float(weights[index]),
                    "organ_family": organ_family(site),
                    "disease_family": disease_family(disease, site, tumor_status),
                    "anatomical_site": site,
                    "tumor_status": tumor_status,
                    "tissue_context": _clean_value(row.get("feat_tissue_context", "unknown")),
                    "biospecimen_type": _clean_value(row.get("feat_biospecimen_type", "unknown")),
                    "disease_label": disease,
                    "prototype_text": _clean_value(row.get("prototype_text", "")),
                }
            )

        leading = components[0] if components else {}
        secondary = [c["disease_family"] for c in components[1:3]]
        secondary_text = ", ".join(secondary) if secondary else "no strong secondary component"
        portrait_text = (
            f"This bulk RNA profile is aligned most strongly with a {leading.get('disease_family', 'unknown')} "
            f"molecular portrait in a {leading.get('organ_family', 'unknown')} context. "
            f"The leading prototype is anchored to {leading.get('anatomical_site', 'unknown')} "
            f"with tumor-status anchor: {leading.get('tumor_status', 'unknown')}. "
            f"Weaker accompanying components include {secondary_text}."
        )

        return {
            "portrait_text": portrait_text,
            "portrait_summary": {
                "organ_family": leading.get("organ_family", "unknown"),
                "disease_family": leading.get("disease_family", "unknown"),
                "tumor_status": leading.get("tumor_status", "unknown"),
                "top_component_weight": float(components[0]["weight"]) if components else 0.0,
                "matched_genes": encoded["matched_genes"],
                "matched_gene_fraction": encoded["matched_gene_fraction"],
                "age_like_years": encoded["age_like_years"],
                "tumor_probability": encoded["tumor_probability"],
            },
            "portrait_components": components,
        }

    def describe_file(self, path: str | Path, top_k: int = 5, log1p: bool = False) -> dict[str, object]:
        return self.describe(read_gene_expression_table(path), top_k=top_k, log1p=log1p)


def load_model(model_root: str | Path | None = None, device: str | torch.device | None = None) -> RNAPortraitModel:
    return RNAPortraitModel.from_pretrained(model_root=model_root, device=device)


def describe_profile(
    expression_values: Mapping[str, float],
    model_root: str | Path | None = None,
    top_k: int = 5,
    log1p: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    model = load_model(model_root=model_root, device=device)
    return model.describe(expression_values, top_k=top_k, log1p=log1p)


def describe_profile_file(
    path: str | Path,
    model_root: str | Path | None = None,
    top_k: int = 5,
    log1p: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    model = load_model(model_root=model_root, device=device)
    return model.describe_file(path, top_k=top_k, log1p=log1p)
