from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EMBED_DIM = 256
DROPOUT = 0.15
TEMPERATURE_INIT = math.log(1 / 0.07)


class BulkRNALanguageAligner(nn.Module):
    """RNA-to-language alignment backbone used in the trained checkpoint."""

    def __init__(self, n_genes: int, n_text_features: int, n_sources: int):
        super().__init__()
        self.gene_gate = nn.Parameter(torch.zeros(n_genes))
        self.expr_encoder = nn.Sequential(
            nn.Linear(n_genes, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, EMBED_DIM),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(n_text_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, EMBED_DIM),
        )
        self.age_head = nn.Linear(EMBED_DIM, 1)
        self.sex_head = nn.Linear(EMBED_DIM, 2)
        self.tumor_head = nn.Linear(EMBED_DIM, 1)
        self.source_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.GELU(),
            nn.Linear(128, n_sources),
        )
        self.logit_scale = nn.Parameter(torch.tensor(TEMPERATURE_INIT))

    def encode_expr(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gene_gate)
        z = self.expr_encoder(x * gate)
        return F.normalize(z, dim=1)

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:
        z = self.text_encoder(x)
        return F.normalize(z, dim=1)


class PortraitAttention(nn.Module):
    """Prototype-attention module that maps an RNA embedding to portrait components."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        hidden_dim: int = 256,
        topk_prototypes: int = 0,
    ):
        super().__init__()
        self.topk_prototypes = int(topk_prototypes)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.proto_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)
        self.post = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor, prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.topk_prototypes > 0 and self.topk_prototypes < prototypes.shape[0]:
            prior_scores = torch.matmul(
                F.normalize(query, dim=1),
                F.normalize(prototypes, dim=1).T,
            )
            topk_idx = torch.topk(prior_scores, k=self.topk_prototypes, dim=1).indices
            proto_bank = prototypes[topk_idx]
        else:
            topk_idx = None
            proto_bank = prototypes.unsqueeze(0).expand(query.shape[0], -1, -1)

        q = self.query_proj(query).unsqueeze(1)
        kv = self.proto_proj(proto_bank)
        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True)

        if topk_idx is not None:
            dense_weights = torch.zeros(
                query.shape[0],
                prototypes.shape[0],
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )
            dense_weights.scatter_(1, topk_idx, attn_weights.squeeze(1))
            attn_weights = dense_weights.unsqueeze(1)

        fused = torch.cat([query, attn_out.squeeze(1)], dim=1)
        recon = F.normalize(self.post(fused), dim=1)
        gate = self.mix_gate(fused)
        mixed = F.normalize(gate * recon + (1.0 - gate) * query, dim=1)
        return mixed, attn_weights.squeeze(1)
