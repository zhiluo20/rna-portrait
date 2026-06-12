#!/usr/bin/env python3
"""Train the age-like adapter required by the unknown-sample explainer.

This script is intentionally small and reproducible. It starts from the
RNA-language alignment backbone trained in the same reproduction run, embeds
training samples with the frozen RNA tower, and fits the lightweight adapter
consumed by `infer_unknown_rna_sample.py`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from repro_paths import BME_CODE_DIR, OUTPUT_ROOT


BASE_RUN = os.getenv("BMM_RUN_NAME", "rna_language_alignment")
RUN_NAME = os.getenv("AGE_ADAPTER_RUN_NAME", "train_age_adapter_curated_ageq_batch18_sbert_allminilm_v2")
HIDDEN_DIM = int(os.getenv("AGE_ADAPTER_HIDDEN_DIM", "128"))
DROPOUT = float(os.getenv("AGE_ADAPTER_DROPOUT", "0.1"))
EPOCHS = int(os.getenv("AGE_ADAPTER_EPOCHS", "160"))
LR = float(os.getenv("AGE_ADAPTER_LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("AGE_ADAPTER_WEIGHT_DECAY", "1e-4"))
SEED = int(os.getenv("AGE_ADAPTER_SEED", "42"))
BATCH_SIZE = int(os.getenv("AGE_ADAPTER_BATCH_SIZE", "2048"))

BASE_CHECKPOINT = OUTPUT_ROOT / BASE_RUN / "bulk_multimodal_embedding.pt"
TRAIN_SCRIPT = BME_CODE_DIR / "train_rna_language_alignment.py"
OUTDIR = OUTPUT_ROOT / RUN_NAME


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("age_adapter_base_train_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AgeAdapter(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def split_age_rows(meta: pd.DataFrame, checkpoint: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = meta.copy()
    meta["feat_age"] = pd.to_numeric(meta["feat_age"], errors="coerce")
    split_assignments = {str(k): str(v) for k, v in checkpoint.get("split_assignments", {}).items()}
    if split_assignments:
        meta["split"] = meta["sample_id"].astype(str).map(split_assignments).fillna("unknown")
    else:
        meta["split"] = "train"
    train = meta.loc[(meta["split"].eq("train")) & meta["feat_age"].notna()].copy()
    val = meta.loc[(meta["split"].eq("val")) & meta["feat_age"].notna()].copy()
    test = meta.loc[(meta["split"].eq("test")) & meta["feat_age"].notna()].copy()
    if val.empty or test.empty:
        age_meta = meta.loc[meta["feat_age"].notna()].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        n = len(age_meta)
        train = age_meta.iloc[: int(0.8 * n)].copy()
        val = age_meta.iloc[int(0.8 * n) : int(0.9 * n)].copy()
        test = age_meta.iloc[int(0.9 * n) :].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def encode_rows(module, model, bundle, checkpoint: dict, rows: pd.DataFrame, device: torch.device) -> np.ndarray:
    selected_genes = list(checkpoint["selected_genes"])
    expr = module.load_expr_subset(bundle, selected_genes, rows["sample_id"].astype(str).tolist())
    x = expr.T.to_numpy(dtype=np.float32)
    mean = np.asarray(checkpoint["expr_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["expr_std"], dtype=np.float32)
    std = np.where(std < 1e-3, 1.0, std)
    x = np.clip((x - mean) / std, -8.0, 8.0)
    outs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], BATCH_SIZE):
            xb = torch.tensor(x[start : start + BATCH_SIZE], dtype=torch.float32, device=device)
            outs.append(model.encode_expr(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def train_adapter(model: AgeAdapter, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> Tuple[AgeAdapter, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("BMM_FORCE_CPU", "0") != "1" else "cpu")
    model = model.to(device)
    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_val = float("inf")
    history = []
    rng = np.random.default_rng(SEED)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = torch.tensor(order[start : start + BATCH_SIZE], dtype=torch.long, device=device)
            pred = model(xtr[idx])
            loss = torch.mean((pred - ytr[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(xva)
            val_loss = float(torch.mean((val_pred - yva) ** 2).cpu())
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu(), {"history": history, "best_val_mse": best_val}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "age_mae_years": float(mean_absolute_error(y_true, y_pred)),
        "age_rmse_years": float(mean_squared_error(y_true, y_pred, squared=False)),
        "age_bucket_acc_3": float(np.mean(np.digitize(y_true, [45, 70]) == np.digitize(y_pred, [45, 70]))),
    }


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    module = load_module(TRAIN_SCRIPT)
    checkpoint = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    module.set_seed(int(checkpoint.get("config", {}).get("seed", SEED)))
    bundle = module.load_dataset()
    meta = bundle.meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    train_meta, val_meta, test_meta = split_age_rows(meta, checkpoint)

    base_model = module.BulkRNALanguageAligner(
        n_genes=len(checkpoint["selected_genes"]),
        n_text_features=checkpoint["config"]["n_text_features"],
        n_sources=len(checkpoint["source_map"]),
    )
    base_model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("BMM_FORCE_CPU", "0") != "1" else "cpu")
    base_model = base_model.to(device).eval()

    x_train = encode_rows(module, base_model, bundle, checkpoint, train_meta, device)
    x_val = encode_rows(module, base_model, bundle, checkpoint, val_meta, device)
    x_test = encode_rows(module, base_model, bundle, checkpoint, test_meta, device)
    y_train = train_meta["feat_age"].to_numpy(dtype=np.float32)
    y_val = val_meta["feat_age"].to_numpy(dtype=np.float32)
    y_test = test_meta["feat_age"].to_numpy(dtype=np.float32)
    y_mean = float(np.nanmean(y_train))
    y_std = float(np.nanstd(y_train) or 1.0)
    y_train_z = (y_train - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    adapter = AgeAdapter(embed_dim=x_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    adapter, train_info = train_adapter(adapter, x_train, y_train_z, x_val, y_val_z)
    with torch.no_grad():
        val_pred = adapter(torch.tensor(x_val, dtype=torch.float32)).numpy() * y_std + y_mean
        test_pred = adapter(torch.tensor(x_test, dtype=torch.float32)).numpy() * y_std + y_mean

    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_hidden_dim": HIDDEN_DIM,
        "adapter_dropout": DROPOUT,
        "base_checkpoint_path": str(BASE_CHECKPOINT),
        "base_train_script": str(TRAIN_SCRIPT),
        "expr_mean": checkpoint["expr_mean"],
        "expr_std": checkpoint["expr_std"],
        "age_mean": y_mean,
        "age_std": y_std,
    }
    torch.save(payload, OUTDIR / "age_adapter.pt")
    pd.DataFrame(train_info["history"]).to_csv(OUTDIR / "training_history.csv", index=False)
    pd.DataFrame({"sample_id": val_meta["sample_id"], "age": y_val, "pred_age": val_pred}).to_csv(OUTDIR / "val_age_predictions.csv", index=False)
    pd.DataFrame({"sample_id": test_meta["sample_id"], "age": y_test, "pred_age": test_pred}).to_csv(OUTDIR / "test_age_predictions.csv", index=False)
    summary = {
        "run_name": RUN_NAME,
        "base_train_run": BASE_RUN,
        "seed": SEED,
        "adapter_hidden_dim": HIDDEN_DIM,
        "adapter_dropout": DROPOUT,
        "train_age_samples": int(len(train_meta)),
        "val_age_samples": int(len(val_meta)),
        "test_age_samples": int(len(test_meta)),
        "best_val_mse_z": float(train_info["best_val_mse"]),
        "val_metrics": metrics(y_val, val_pred),
        "test_metrics": metrics(y_test, test_pred),
    }
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUTDIR / "summary.md").write_text(
        "\n".join(
            [
                "# Age-like Adapter",
                "",
                f"- train_age_samples: `{summary['train_age_samples']}`",
                f"- val_age_samples: `{summary['val_age_samples']}`",
                f"- test_age_samples: `{summary['test_age_samples']}`",
                f"- val_age_mae_years: `{summary['val_metrics']['age_mae_years']:.4f}`",
                f"- test_age_mae_years: `{summary['test_metrics']['age_mae_years']:.4f}`",
            ]
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
