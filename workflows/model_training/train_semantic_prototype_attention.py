#!/usr/bin/env python3
"""Train a semantic prototype-bank attention explainer on top of a semantic RNA-text run."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from repro_paths import BME_CODE_DIR as MODULE_DIR
from repro_paths import OUTPUT_ROOT


ROOT = Path(__file__).resolve().parents[2]

BASE_RUN = os.getenv("SPTA_BASE_RUN", "rna_language_alignment")
TRAIN_SCRIPT = MODULE_DIR / "train_rna_language_alignment.py"

RUN_NAME = os.getenv("SPTA_RUN_NAME", "semantic_backbone_v8_topk64")
OUTDIR = OUTPUT_ROOT / RUN_NAME
TEXT_SOURCE = os.getenv("SPTA_TEXT_SOURCE", "caption_text").strip()

N_PROTOTYPES = int(os.getenv("SPTA_N_PROTOTYPES", "128"))
N_HEADS = int(os.getenv("SPTA_N_HEADS", "8"))
HIDDEN_DIM = int(os.getenv("SPTA_HIDDEN_DIM", "256"))
EPOCHS = int(os.getenv("SPTA_EPOCHS", "30"))
PATIENCE = int(os.getenv("SPTA_PATIENCE", "6"))
BATCH_SIZE = int(os.getenv("SPTA_BATCH_SIZE", "512"))
LR = float(os.getenv("SPTA_LR", "8e-4"))
WEIGHT_DECAY = float(os.getenv("SPTA_WEIGHT_DECAY", "1e-4"))
TARGET_TEMP = float(os.getenv("SPTA_TARGET_TEMP", "0.08"))
KL_WEIGHT = float(os.getenv("SPTA_KL_WEIGHT", "0.35"))
RETRIEVAL_WEIGHT = float(os.getenv("SPTA_RETRIEVAL_WEIGHT", "0.35"))
CLIP_TEMP = float(os.getenv("SPTA_CLIP_TEMP", "0.07"))
TOPK_PROTOTYPES = int(os.getenv("SPTA_TOPK_PROTOTYPES", "0"))


def resolve_base_run_dir() -> Path:
    preferred = OUTPUT_ROOT / BASE_RUN
    if preferred.exists():
        return preferred
    for fallback_name in ["rna_language_alignment", "semantic_alignment_backbone"]:
        fallback = OUTPUT_ROOT / fallback_name
        if fallback.exists():
            return fallback
    return preferred


BASE_RUN_DIR = resolve_base_run_dir()
BASE_CHECKPOINT = BASE_RUN_DIR / "bulk_multimodal_embedding.pt"
BASE_TEXT_CACHE = BASE_RUN_DIR / "caption_text_embeddings.npy"


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("semantic_proto_train_base", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_checkpoint() -> dict:
    return torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return x / norms


def build_semantic_core_text(text: str) -> str:
    t = str(text).strip()
    t = re.sub(r"\b\d{1,3}-year-old\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:male|female)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:sample|specimen|case)\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:project|sample id)\s+[A-Za-z0-9._:-]+\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,+", ", ", t)
    return t.strip(" ,.;")


def topk_recall(sim: np.ndarray, ks: List[int]) -> Dict[str, float]:
    if sim.shape[0] == 0:
        return {f"r@{k}": float("nan") for k in ks}
    max_k = min(max(ks), sim.shape[1])
    idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    row = np.arange(sim.shape[0])[:, None]
    scores = sim[row, idx]
    order = np.argsort(-scores, axis=1)
    ranked = idx[row, order]
    truth = np.arange(sim.shape[0])
    out = {}
    for k in ks:
        kk = min(k, ranked.shape[1])
        out[f"r@{k}"] = float(np.mean([truth[i] in ranked[i, :kk] for i in range(sim.shape[0])]))
    return out


@dataclass
class SemanticArtifacts:
    module: object
    checkpoint: dict
    model: torch.nn.Module
    bundle: object
    meta: pd.DataFrame
    selected_genes: List[str]
    expr_mean: np.ndarray
    expr_std: np.ndarray
    text_raw: np.ndarray
    text_source: str
    semantic_text_column: str


class SemanticPrototypeAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int = N_HEADS, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
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
        if TOPK_PROTOTYPES > 0 and TOPK_PROTOTYPES < prototypes.shape[0]:
            prior_scores = torch.matmul(
                F.normalize(query, dim=1),
                F.normalize(prototypes, dim=1).T,
            )
            topk_idx = torch.topk(prior_scores, k=TOPK_PROTOTYPES, dim=1).indices
            proto_bank = prototypes[topk_idx]
        else:
            proto_bank = prototypes.unsqueeze(0).expand(query.shape[0], -1, -1)
        q = self.query_proj(query).unsqueeze(1)
        kv = self.proto_proj(proto_bank)
        attn_out, attn_weights = self.attn(q, kv, kv, need_weights=True)
        if TOPK_PROTOTYPES > 0 and TOPK_PROTOTYPES < prototypes.shape[0]:
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


def build_base_model(module, checkpoint: dict) -> torch.nn.Module:
    model = module.BulkRNALanguageAligner(
        n_genes=len(checkpoint["selected_genes"]),
        n_text_features=checkpoint["config"]["n_text_features"],
        n_sources=len(checkpoint["source_map"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_artifacts() -> SemanticArtifacts:
    module = load_training_module(TRAIN_SCRIPT)
    checkpoint = load_checkpoint()
    module.set_seed(int(checkpoint.get("config", {}).get("seed", getattr(module, "SEED", 42))))
    model = build_base_model(module, checkpoint)
    bundle = module.load_dataset()
    meta = bundle.meta.copy()
    meta["caption_text"] = meta.apply(module.build_caption, axis=1)
    meta["sample_id"] = meta["sample_id"].astype(str)
    if TEXT_SOURCE == "semantic_core_text":
        meta["semantic_core_text"] = meta["metadata_text"].astype(str).map(build_semantic_core_text)
        cache_path = OUTDIR / "semantic_core_text_embeddings.npy"
        text_model = module.load_text_model()
        text_raw = module.encode_text_corpus(text_model, meta["semantic_core_text"].astype(str).tolist(), cache_path).astype(np.float32)
        semantic_text_column = "semantic_core_text"
    elif TEXT_SOURCE == "metadata_text":
        cache_path = OUTDIR / "metadata_text_embeddings.npy"
        text_model = module.load_text_model()
        text_raw = module.encode_text_corpus(text_model, meta["metadata_text"].astype(str).tolist(), cache_path).astype(np.float32)
        semantic_text_column = "metadata_text"
    else:
        text_raw = np.load(BASE_TEXT_CACHE).astype(np.float32)
        semantic_text_column = "caption_text"
    return SemanticArtifacts(
        module=module,
        checkpoint=checkpoint,
        model=model,
        bundle=bundle,
        meta=meta,
        selected_genes=list(checkpoint["selected_genes"]),
        expr_mean=np.asarray(checkpoint["expr_mean"], dtype=np.float32),
        expr_std=np.where(np.asarray(checkpoint["expr_std"], dtype=np.float32) < 1e-3, 1.0, np.asarray(checkpoint["expr_std"], dtype=np.float32)),
        text_raw=text_raw,
        text_source=TEXT_SOURCE,
        semantic_text_column=semantic_text_column,
    )


def build_splits(art: SemanticArtifacts) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ck = art.checkpoint
    if ck.get("split_assignments"):
        split_assignments = {str(k): str(v) for k, v in ck["split_assignments"].items()}
        meta = art.meta.copy()
        meta["split"] = meta["sample_id"].map(split_assignments).fillna("unknown")
        train_meta = meta.loc[meta["split"] == "train"].copy().reset_index(drop=True)
        val_meta = meta.loc[meta["split"] == "val"].copy().reset_index(drop=True)
        test_meta = meta.loc[meta["split"] == "test"].copy().reset_index(drop=True)
        return train_meta, val_meta, test_meta
    train_meta, val_meta, test_meta = art.module.build_splits(art.meta.copy())
    return train_meta.reset_index(drop=True), val_meta.reset_index(drop=True), test_meta.reset_index(drop=True)


def encode_expr_embeddings(art: SemanticArtifacts, meta: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
    expr = art.module.load_expr_subset(art.bundle, art.selected_genes, meta["sample_id"].astype(str).tolist())
    x = expr.T.to_numpy(dtype=np.float32)
    x = np.clip((x - art.expr_mean) / art.expr_std, -8.0, 8.0)
    embs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
            z = art.model.encode_expr(xb)
            embs.append(z.cpu().numpy())
    return np.vstack(embs).astype(np.float32)


def encode_text_embeddings(art: SemanticArtifacts, sample_ids: List[str], batch_size: int = 2048) -> np.ndarray:
    full_ids = art.meta["sample_id"].astype(str).tolist()
    row_index = {sid: i for i, sid in enumerate(full_ids)}
    rows = [row_index[str(sid)] for sid in sample_ids]
    x = art.text_raw[rows].astype(np.float32)
    embs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
            z = art.model.encode_text(xb)
            embs.append(z.cpu().numpy())
    return np.vstack(embs).astype(np.float32)


def build_target_proto_dist(text_emb: np.ndarray, proto_emb: np.ndarray, temp: float = TARGET_TEMP) -> np.ndarray:
    sims = normalize_rows(text_emb) @ normalize_rows(proto_emb).T
    sims = sims / temp
    sims = sims - sims.max(axis=1, keepdims=True)
    probs = np.exp(sims)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    return probs.astype(np.float32)


def representative_rows(train_text_emb: np.ndarray, train_meta: pd.DataFrame, proto_emb: np.ndarray, semantic_text_column: str) -> pd.DataFrame:
    sims = normalize_rows(proto_emb) @ normalize_rows(train_text_emb).T
    best_idx = sims.argmax(axis=1)
    rep = train_meta.iloc[best_idx].copy().reset_index(drop=True)
    rep["prototype_id"] = np.arange(len(rep))
    rep["prototype_size"] = 0
    rep["prototype_semantic_text"] = rep[semantic_text_column].astype(str)
    return rep


def fit_prototypes(train_text_emb: np.ndarray, train_meta: pd.DataFrame, semantic_text_column: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    km = MiniBatchKMeans(n_clusters=N_PROTOTYPES, random_state=42, batch_size=2048, n_init=10)
    labels = km.fit_predict(train_text_emb)
    proto_emb = normalize_rows(km.cluster_centers_.astype(np.float32))
    rep = representative_rows(train_text_emb, train_meta, proto_emb, semantic_text_column)
    counts = pd.Series(labels).value_counts().to_dict()
    rep["prototype_size"] = rep["prototype_id"].map(counts).fillna(0).astype(int)
    return proto_emb, rep, labels.astype(np.int64)


def tensor_batches(xq: np.ndarray, yt: np.ndarray, yp: np.ndarray, batch_size: int = BATCH_SIZE):
    idx = np.arange(xq.shape[0])
    np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        take = idx[start:start + batch_size]
        yield (
            torch.tensor(xq[take], dtype=torch.float32),
            torch.tensor(yt[take], dtype=torch.float32),
            torch.tensor(yp[take], dtype=torch.float32),
        )


def evaluate(model: SemanticPrototypeAttention, x_expr: np.ndarray, y_text: np.ndarray, y_proto: np.ndarray, proto_bank: np.ndarray) -> dict:
    model.eval()
    proto_t = torch.tensor(proto_bank, dtype=torch.float32)
    recon_chunks = []
    weight_chunks = []
    with torch.no_grad():
        for start in range(0, x_expr.shape[0], 1024):
            xb = torch.tensor(x_expr[start:start + 1024], dtype=torch.float32)
            mixed, weights = model(xb, proto_t)
            recon_chunks.append(mixed.cpu().numpy())
            weight_chunks.append(weights.cpu().numpy())
    recon = np.vstack(recon_chunks).astype(np.float32)
    weights = np.vstack(weight_chunks).astype(np.float32)
    cosine_self = np.sum(normalize_rows(recon) * normalize_rows(y_text), axis=1)
    retrieval = normalize_rows(recon) @ normalize_rows(y_text).T
    metrics = {
        "semantic_self_cosine": float(np.mean(cosine_self)),
        "proto_entropy": float(np.mean((-weights * np.log(np.clip(weights, 1e-8, None))).sum(axis=1))),
        **topk_recall(retrieval, [1, 5, 10]),
    }
    metrics["proto_top1_match"] = float(np.mean(weights.argmax(axis=1) == y_proto.argmax(axis=1)))
    return metrics


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    art = load_artifacts()
    train_meta, val_meta, test_meta = build_splits(art)

    train_expr = encode_expr_embeddings(art, train_meta)
    val_expr = encode_expr_embeddings(art, val_meta)
    test_expr = encode_expr_embeddings(art, test_meta)

    train_text = encode_text_embeddings(art, train_meta["sample_id"].tolist())
    val_text = encode_text_embeddings(art, val_meta["sample_id"].tolist())
    test_text = encode_text_embeddings(art, test_meta["sample_id"].tolist())

    proto_bank, proto_table, train_proto_labels = fit_prototypes(train_text, train_meta, art.semantic_text_column)
    val_proto = build_target_proto_dist(val_text, proto_bank)
    test_proto = build_target_proto_dist(test_text, proto_bank)
    train_proto = build_target_proto_dist(train_text, proto_bank)

    model = SemanticPrototypeAttention(embed_dim=train_expr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_score = None
    bad_epochs = 0
    proto_t = torch.tensor(proto_bank, dtype=torch.float32)

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb, pb in tensor_batches(train_expr, train_text, train_proto):
            mixed, weights = model(xb, proto_t)
            cos_loss = (1.0 - F.cosine_similarity(mixed, yb, dim=1)).mean()
            kl_loss = F.kl_div(torch.log(torch.clamp(weights, 1e-8, None)), pb, reduction="batchmean")
            logits = torch.matmul(mixed, yb.T) / CLIP_TEMP
            ret_loss = clip_loss(logits)
            loss = cos_loss + KL_WEIGHT * kl_loss + RETRIEVAL_WEIGHT * ret_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate(model, val_expr, val_text, val_proto, proto_bank)
        score = float(val_metrics["semantic_self_cosine"] + 0.5 * val_metrics["r@1"] + 0.2 * val_metrics["r@5"])
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_expr, train_text, train_proto, proto_bank)
    val_metrics = evaluate(model, val_expr, val_text, val_proto, proto_bank)
    test_metrics = evaluate(model, test_expr, test_text, test_proto, proto_bank)

    model.eval()
    with torch.no_grad():
        recon_test, weights_test = model(torch.tensor(test_expr, dtype=torch.float32), proto_t)
    recon_test_np = recon_test.cpu().numpy().astype(np.float32)
    weights_test_np = weights_test.cpu().numpy().astype(np.float32)
    top_proto = weights_test_np.argmax(axis=1)
    top_weight = weights_test_np[np.arange(len(top_proto)), top_proto]

    test_out = test_meta.copy()
    test_out["semantic_self_cosine"] = np.sum(normalize_rows(recon_test_np) * normalize_rows(test_text), axis=1)
    test_out["top_prototype_id"] = top_proto
    test_out["top_prototype_weight"] = top_weight
    test_out["top_prototype_caption"] = [str(proto_table.iloc[i]["prototype_semantic_text"]) for i in top_proto]
    test_out["top_prototype_site_anchor"] = [str(proto_table.iloc[i]["feat_anatomical_site"]) for i in top_proto]
    test_out["top_prototype_tumor_anchor"] = [str(proto_table.iloc[i]["feat_tumor_status"]) for i in top_proto]
    test_out["top_prototype_disease_anchor"] = [str(proto_table.iloc[i]["feat_disease_label"]) for i in top_proto]

    summary = {
        "base_run": BASE_RUN,
        "text_source": art.text_source,
        "semantic_text_column": art.semantic_text_column,
        "n_train": int(len(train_meta)),
        "n_val": int(len(val_meta)),
        "n_test": int(len(test_meta)),
        "n_prototypes": int(len(proto_bank)),
        "n_heads": int(N_HEADS),
        "topk_prototypes": int(TOPK_PROTOTYPES),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    proto_table.to_csv(OUTDIR / "prototype_table.csv", index=False)
    test_out.to_csv(OUTDIR / "test_semantic_predictions.csv", index=False)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "prototype_bank": proto_bank,
            "prototype_table_path": str(OUTDIR / "prototype_table.csv"),
            "summary": summary,
        },
        OUTDIR / "semantic_prototype_attention.pt",
    )
    lines = [
        "# Semantic Prototype Attention",
        "",
        f"- base_run: `{BASE_RUN}`",
        f"- text_source: `{art.text_source}`",
        f"- n_prototypes: `{len(proto_bank)}`",
        f"- n_heads: `{N_HEADS}`",
        f"- topk_prototypes: `{TOPK_PROTOTYPES}`",
        f"- test semantic_self_cosine: `{test_metrics['semantic_self_cosine']:.4f}`",
        f"- test r@1: `{test_metrics['r@1']:.4f}`",
        f"- test r@5: `{test_metrics['r@5']:.4f}`",
        f"- test r@10: `{test_metrics['r@10']:.4f}`",
        f"- test proto_top1_match: `{test_metrics['proto_top1_match']:.4f}`",
        "",
        "This benchmark is semantic-first: primary metrics use caption-space reconstruction and retrieval, while site/tumor/disease remain prototype anchors only.",
    ]
    (OUTDIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
