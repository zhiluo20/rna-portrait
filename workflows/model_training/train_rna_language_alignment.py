#!/usr/bin/env python3
"""Bulk RNA-language semantic alignment model.

This is a lightweight local adaptation for the current project:
- Transcriptome tower: bulk RNA log-expression -> embedding
- Text tower: metadata_text + structured tags -> embedding
- Joint training: symmetric InfoNCE / CLIP-style alignment
- Leakage control: GSE source-adversarial head
- Auxiliary supervision: age, sex, tumor status
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.autograd import Function
from torch.utils.data import Dataset

from device_runtime import select_torch_device
from repro_paths import OUTPUT_ROOT, TRAINING_DATASET_DIR

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv('BMM_DATASET_DIR', str(TRAINING_DATASET_DIR)))
RUN_NAME = os.getenv('BMM_RUN_NAME', 'rna_language_alignment')
TEXT_MODEL_NAME = os.getenv('BMM_TEXT_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
TEXT_MODEL_REVISION = os.getenv(
    'BMM_TEXT_MODEL_REVISION',
    'c9745ed1d9f207416be6d2e6f8de32d1f16199bf',
)
OUTDIR = OUTPUT_ROOT / RUN_NAME
SPLIT_MANIFEST = os.getenv('BMM_SPLIT_MANIFEST', '').strip()
PLOTLY_CDN = 'https://cdn.plot.ly/plotly-2.35.2.min.js'

SEED = int(os.getenv('BMM_SEED', '42'))
N_FEATURES = 4096
MAX_TEXT_FEATURES = 4096
EMBED_DIM = 256
MAX_EPOCHS = 35
PATIENCE = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
AUX_AGE_WEIGHT = 0.15
AUX_SEX_WEIGHT = 0.10
AUX_TUMOR_WEIGHT = 0.10
ADV_WEIGHT = 0.10
TEMPERATURE_INIT = math.log(1 / 0.07)
SITE_POS_WEIGHT = 0.08
DISEASE_POS_WEIGHT = 0.05


def _env_int(name: str) -> Optional[int]:
    value = os.getenv(name)
    return int(value) if value is not None else None


def get_total_memory_gb() -> float:
    try:
        with open('/proc/meminfo', 'r', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('MemTotal:'):
                    parts = line.split()
                    return float(parts[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def get_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def choose_cpu_threads() -> int:
    return _env_int('BMM_CPU_THREADS') or get_cpu_count()


def choose_cpu_interop_threads(cpu_threads: int) -> int:
    return _env_int('BMM_CPU_INTEROP_THREADS') or max(2, min(8, cpu_threads // 8 or 1))


def choose_io_workers(cpu_threads: int) -> int:
    override = _env_int('BMM_NUM_WORKERS')
    if override is not None:
        return max(0, override)
    if cpu_threads <= 4:
        return 2
    return min(24, max(8, cpu_threads - 4))


def choose_loader_workers(device_type: str) -> int:
    override = _env_int('BMM_LOADER_WORKERS')
    if override is not None:
        return max(0, override)
    if device_type == 'cuda':
        return 4
    return 0


def choose_prefetch_factor(num_workers: int) -> int:
    override = _env_int('BMM_PREFETCH_FACTOR')
    if override is not None:
        return max(2, override)
    if num_workers >= 16:
        return 6
    if num_workers >= 8:
        return 4
    return 2


def choose_batch_size(device_type: str, eval_mode: bool) -> int:
    if device_type == 'cuda':
        train_override = _env_int('BMM_BATCH_SIZE_GPU')
        eval_override = _env_int('BMM_EVAL_BATCH_SIZE_GPU')
        if eval_mode and eval_override is not None:
            return eval_override
        if not eval_mode and train_override is not None:
            return train_override
        return 6144 if eval_mode else 3072

    train_override = _env_int('BMM_BATCH_SIZE_CPU')
    eval_override = _env_int('BMM_EVAL_BATCH_SIZE_CPU')
    if eval_mode and eval_override is not None:
        return eval_override
    if not eval_mode and train_override is not None:
        return train_override

    mem_gb = get_total_memory_gb()
    if mem_gb >= 48:
        return 2048 if eval_mode else 1024
    if mem_gb >= 24:
        return 1024 if eval_mode else 512
    return 512 if eval_mode else 256


CPU_THREADS = choose_cpu_threads()
CPU_INTEROP_THREADS = choose_cpu_interop_threads(CPU_THREADS)
IO_WORKERS = choose_io_workers(CPU_THREADS)
LOADER_WORKERS_CPU = choose_loader_workers('cpu')
LOADER_WORKERS_GPU = choose_loader_workers('cuda')
PREFETCH_FACTOR = choose_prefetch_factor(max(LOADER_WORKERS_CPU, LOADER_WORKERS_GPU))
BATCH_SIZE_CPU = choose_batch_size('cpu', eval_mode=False)
EVAL_BATCH_SIZE_CPU = choose_batch_size('cpu', eval_mode=True)
BATCH_SIZE_GPU = choose_batch_size('cuda', eval_mode=False)
EVAL_BATCH_SIZE_GPU = choose_batch_size('cuda', eval_mode=True)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(max(1, CPU_THREADS))
    try:
        torch.set_num_interop_threads(max(1, CPU_INTEROP_THREADS))
    except RuntimeError:
        pass


DEVICE_PROBE = None


def get_device() -> torch.device:
    global DEVICE_PROBE
    if DEVICE_PROBE is None:
        device, probe = select_torch_device()
        DEVICE_PROBE = probe
        return device
    return torch.device(str(DEVICE_PROBE.get('device_type', 'cpu')))


class MultiModalDataset(Dataset):
    def __init__(
        self,
        x_expr: np.ndarray,
        x_text: np.ndarray,
        age_target: np.ndarray,
        age_mask: np.ndarray,
        sex_target: np.ndarray,
        sex_mask: np.ndarray,
        tumor_target: np.ndarray,
        tumor_mask: np.ndarray,
        source_target: np.ndarray,
        sample_ids: np.ndarray,
    ):
        self.x_expr = torch.tensor(x_expr, dtype=torch.float32)
        self.x_text = torch.tensor(x_text, dtype=torch.float32)
        self.age_target = torch.tensor(age_target, dtype=torch.float32)
        self.age_mask = torch.tensor(age_mask, dtype=torch.float32)
        self.sex_target = torch.tensor(sex_target, dtype=torch.long)
        self.sex_mask = torch.tensor(sex_mask, dtype=torch.float32)
        self.tumor_target = torch.tensor(tumor_target, dtype=torch.float32)
        self.tumor_mask = torch.tensor(tumor_mask, dtype=torch.float32)
        self.source_target = torch.tensor(source_target, dtype=torch.long)
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        return (
            self.x_expr[idx],
            self.x_text[idx],
            self.age_target[idx],
            self.age_mask[idx],
            self.sex_target[idx],
            self.sex_mask[idx],
            self.tumor_target[idx],
            self.tumor_mask[idx],
            self.source_target[idx],
            self.sample_ids[idx],
        )


class BulkRNALanguageAligner(nn.Module):
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

    def forward(self, x_expr: torch.Tensor, x_text: torch.Tensor, adv_lambda: float = 1.0):
        z_expr = self.encode_expr(x_expr)
        z_text = self.encode_text(x_text)
        age_pred = self.age_head(z_expr).squeeze(1)
        sex_logit = self.sex_head(z_expr)
        tumor_logit = self.tumor_head(z_expr).squeeze(1)
        source_logit = self.source_head(grad_reverse(z_expr, adv_lambda))
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = torch.matmul(z_expr, z_text.T) * scale
        return z_expr, z_text, logits, age_pred, sex_logit, tumor_logit, source_logit


@dataclass
class SplitData:
    x_expr: np.ndarray
    x_text: np.ndarray
    age_target: np.ndarray
    age_mask: np.ndarray
    sex_target: np.ndarray
    sex_mask: np.ndarray
    tumor_target: np.ndarray
    tumor_mask: np.ndarray
    source_target: np.ndarray
    site_target: np.ndarray
    disease_target: np.ndarray
    meta: pd.DataFrame


@dataclass
class TrainArtifacts:
    model: BulkRNALanguageAligner
    text_backend: str
    text_encoder_name: str
    selected_genes: List[str]
    expr_mean: np.ndarray
    expr_std: np.ndarray
    age_mean: float
    age_std: float
    source_map: Dict[str, int]
    history: List[dict]
    best_epoch: int
    best_val_loss: float
    train_seconds: float
    split_assignments: Dict[str, str]


@dataclass
class DatasetBundle:
    meta: pd.DataFrame
    genes: List[str]
    expr: Optional[pd.DataFrame] = None
    sample_to_shard: Optional[Dict[str, Path]] = None
    layout: str = 'monolithic'


QUERY_LIBRARY = [
    'a healthy non-tumor whole blood sample from an adult individual',
    'a tumor blood sample with leukemic blasts',
    'a prostate cancer cell line sample from a male individual',
    'a non-tumor liver sample from NAFLD tissue',
    'an old male tumor sample with aggressive disease',
    'a female non-tumor blood immune sample',
]


def load_text_model() -> SentenceTransformer:
    return SentenceTransformer(
        TEXT_MODEL_NAME,
        revision=TEXT_MODEL_REVISION,
        device='cpu',
    )


def encode_text_corpus(model: SentenceTransformer, texts: List[str], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached.astype(np.float32)
    emb = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    emb = emb.astype(np.float32)
    np.save(cache_path, emb)
    return emb


def encode_text_queries_backend(model: SentenceTransformer, queries: List[str]) -> np.ndarray:
    return model.encode(queries, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)


def load_monolithic_dataset() -> DatasetBundle:
    expr = pd.read_parquet(DATA_DIR / 'expr_log.parquet')
    meta = pd.read_parquet(DATA_DIR / 'meta.parquet').copy()
    sample_ids = np.load(DATA_DIR / 'sample_ids.npy', allow_pickle=True)
    if len(sample_ids) != expr.shape[1] or len(sample_ids) != len(meta):
        raise ValueError('sample_ids/meta/expr shapes do not match')
    meta['sample_id'] = sample_ids
    if list(expr.columns) != list(sample_ids):
        expr.columns = sample_ids
    expr.index = expr.index.astype(str).str.replace(r'\.\d+$', '', regex=True).str.upper()
    expr = expr.groupby(level=0).mean()
    meta = meta.loc[meta['sample_id'].isin(expr.columns)].copy()
    meta = meta.reset_index(drop=True)
    expr = expr.loc[:, meta['sample_id'].tolist()]
    return DatasetBundle(meta=meta, genes=expr.index.astype(str).tolist(), expr=expr, layout='monolithic')


def load_sharded_dataset() -> DatasetBundle:
    meta = pd.read_parquet(DATA_DIR / 'meta.parquet').copy()
    sample_ids = np.load(DATA_DIR / 'sample_ids.npy', allow_pickle=True)
    genes = np.load(DATA_DIR / 'genes.npy', allow_pickle=True).astype(str).tolist()
    if len(sample_ids) != len(meta):
        raise ValueError('sample_ids/meta shapes do not match for sharded dataset')
    if 'sample_id' not in meta.columns:
        meta['sample_id'] = sample_ids
    sample_to_shard: Dict[str, Path] = {}
    for shard_meta_path in sorted((DATA_DIR / 'projects').glob('*/shard_*/meta.parquet')):
        shard_meta = pd.read_parquet(shard_meta_path, columns=['sample_id'])
        expr_path = shard_meta_path.with_name('expr_log.parquet')
        for sample_id in shard_meta['sample_id'].astype(str).tolist():
            sample_to_shard[sample_id] = expr_path
    missing = [sid for sid in meta['sample_id'].astype(str).tolist() if sid not in sample_to_shard]
    if missing:
        raise ValueError(f'missing shard mapping for {len(missing)} samples')
    return DatasetBundle(meta=meta.reset_index(drop=True), genes=genes, sample_to_shard=sample_to_shard, layout='sharded')


def load_dataset() -> DatasetBundle:
    if (DATA_DIR / 'expr_log.parquet').exists():
        return load_monolithic_dataset()
    if (DATA_DIR / 'projects').exists():
        return load_sharded_dataset()
    raise FileNotFoundError(f'Unsupported dataset layout under {DATA_DIR}')


def safe_str(v: object, default: str = 'unknown') -> str:
    if pd.isna(v):
        return default
    s = str(v).strip()
    return s if s else default


def build_caption(row: pd.Series) -> str:
    free = safe_str(row.get('metadata_text'), '')
    parts = []
    age = row.get('feat_age')
    if pd.notna(age):
        parts.append(f'age {int(round(float(age)))}')
    parts.append(f'sex {safe_str(row.get("feat_sex"))}')
    parts.append(f'biospecimen {safe_str(row.get("feat_biospecimen_type"))}')
    parts.append(f'anatomical site {safe_str(row.get("feat_anatomical_site"))}')
    parts.append(f'tumor status {safe_str(row.get("feat_tumor_status"))}')
    parts.append(f'tissue context {safe_str(row.get("feat_tissue_context"))}')
    parts.append(f'disease {safe_str(row.get("feat_disease_label"))}')
    parts.append(f'disease severity {safe_str(row.get("feat_disease_severity"))}')
    parts.append(f'sample role {safe_str(row.get("feat_sample_role"))}')
    parts.append(f'gse {safe_str(row.get("gse"))}')
    structured = 'Structured metadata: ' + '; '.join(parts) + '.'
    if free and free.lower() != 'unknown':
        return free + ' ' + structured
    return structured


def coarse_tumor(v: object) -> Tuple[float, float]:
    s = safe_str(v)
    if s in {'tumor', 'metastatic'}:
        return 1.0, 1.0
    if s in {'non_tumor', 'adjacent_normal', 'not_applicable'}:
        return 0.0, 1.0
    return 0.0, 0.0


def sex_label(v: object) -> Tuple[int, float]:
    s = safe_str(v)
    if s == 'male':
        return 0, 1.0
    if s == 'female':
        return 1, 1.0
    return 0, 0.0


def select_hvg(expr: pd.DataFrame, train_ids: List[str], n_features: int = N_FEATURES) -> List[str]:
    sub = expr.loc[:, train_ids]
    mean = sub.mean(axis=1)
    var = sub.var(axis=1, ddof=0)
    detect = (sub > 0.1).mean(axis=1)
    dispersion = (var / (mean + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = pd.DataFrame({'gene': sub.index, 'mean': mean, 'var': var, 'detect': detect, 'dispersion': dispersion})
    df = df[(df['detect'] >= 0.1) & (df['var'] > 0.02)].sort_values('dispersion', ascending=False)
    return df.head(n_features)['gene'].tolist()


def select_hvg_sharded(bundle: DatasetBundle, train_ids: List[str], n_features: int = N_FEATURES) -> List[str]:
    if bundle.sample_to_shard is None:
        raise ValueError('sharded bundle missing sample_to_shard index')
    train_ids = [str(x) for x in train_ids]
    genes = np.array(bundle.genes, dtype=object)
    n_genes = len(genes)
    sum_x = np.zeros(n_genes, dtype=np.float64)
    sum_x2 = np.zeros(n_genes, dtype=np.float64)
    detect = np.zeros(n_genes, dtype=np.float64)
    n_samples = 0

    shard_to_ids: Dict[Path, List[str]] = {}
    for sid in train_ids:
        shard_to_ids.setdefault(bundle.sample_to_shard[sid], []).append(sid)

    def shard_stats(item: Tuple[Path, List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        shard_path, shard_ids = item
        expr = pd.read_parquet(shard_path)
        expr = expr.loc[:, shard_ids]
        arr = expr.to_numpy(dtype=np.float64, copy=False)
        return (
            arr.sum(axis=1),
            np.square(arr).sum(axis=1),
            (arr > 0.1).sum(axis=1).astype(np.float64),
            int(arr.shape[1]),
        )

    items = list(shard_to_ids.items())
    max_workers = min(IO_WORKERS or 1, len(items)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for shard_sum, shard_sum2, shard_detect, shard_n in ex.map(shard_stats, items):
            sum_x += shard_sum
            sum_x2 += shard_sum2
            detect += shard_detect
            n_samples += shard_n

    mean = sum_x / max(n_samples, 1)
    var = np.maximum(sum_x2 / max(n_samples, 1) - np.square(mean), 0.0)
    detect_frac = detect / max(n_samples, 1)
    dispersion = np.divide(var, mean + 1.0, out=np.zeros_like(var), where=np.isfinite(var))
    df = pd.DataFrame({'gene': genes, 'mean': mean, 'var': var, 'detect': detect_frac, 'dispersion': dispersion})
    df = df[(df['detect'] >= 0.1) & (df['var'] > 0.02)].sort_values('dispersion', ascending=False)
    return df.head(n_features)['gene'].astype(str).tolist()


def load_expr_subset(bundle: DatasetBundle, genes: List[str], sample_ids: List[str]) -> pd.DataFrame:
    sample_ids = [str(x) for x in sample_ids]
    if bundle.layout == 'monolithic':
        if bundle.expr is None:
            raise ValueError('monolithic bundle missing expr matrix')
        return bundle.expr.loc[genes, sample_ids]

    if bundle.sample_to_shard is None:
        raise ValueError('sharded bundle missing sample_to_shard index')

    gene_index = pd.Index(genes)
    parts = []
    shard_to_ids: Dict[Path, List[str]] = {}
    for sid in sample_ids:
        shard_to_ids.setdefault(bundle.sample_to_shard[sid], []).append(sid)

    def read_part(item: Tuple[Path, List[str]]) -> pd.DataFrame:
        shard_path, shard_ids = item
        expr = pd.read_parquet(shard_path)
        return expr.loc[gene_index, shard_ids]

    items = list(shard_to_ids.items())
    max_workers = min(IO_WORKERS or 1, len(items)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        parts = list(ex.map(read_part, items))
    out = pd.concat(parts, axis=1)
    return out.loc[gene_index, sample_ids]


def standardize_expr(x_train: np.ndarray, x_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-3, 1.0, std)
    x_train_out = np.clip((x_train - mean) / std, -8.0, 8.0)
    x_other_out = np.clip((x_other - mean) / std, -8.0, 8.0)
    return x_train_out, x_other_out, mean, std


def build_splits(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if SPLIT_MANIFEST:
        payload = json.loads(Path(SPLIT_MANIFEST).read_text(encoding='utf-8'))
        sample_to_split = payload.get('sample_to_split', payload)
        split_series = meta['sample_id'].astype(str).map(sample_to_split)
        if split_series.isna().any():
            missing = meta.loc[split_series.isna(), 'sample_id'].astype(str).tolist()[:10]
            raise ValueError(f'split manifest missing assignments for samples like: {missing}')
        train = meta.loc[split_series == 'train'].copy().reset_index(drop=True)
        val = meta.loc[split_series == 'val'].copy().reset_index(drop=True)
        test = meta.loc[split_series == 'test'].copy().reset_index(drop=True)
        if train.empty or val.empty or test.empty:
            raise ValueError(f'invalid split manifest at {SPLIT_MANIFEST}')
        return train, val, test

    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED)
    idx = np.arange(len(meta))
    train_val_idx, test_idx = next(gss_outer.split(idx, groups=meta['gse']))
    train_val = meta.iloc[train_val_idx].copy().reset_index(drop=True)
    test = meta.iloc[test_idx].copy().reset_index(drop=True)

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED + 1)
    idx_tv = np.arange(len(train_val))
    train_idx, val_idx = next(gss_inner.split(idx_tv, groups=train_val['gse']))
    train = train_val.iloc[train_idx].copy().reset_index(drop=True)
    val = train_val.iloc[val_idx].copy().reset_index(drop=True)
    return train, val, test


def prepare_split_data(
    expr: pd.DataFrame,
    split_meta: pd.DataFrame,
    genes: List[str],
    text_matrix: np.ndarray,
    text_row_index: Dict[str, int],
    expr_mean: np.ndarray,
    expr_std: np.ndarray,
    age_mean: float,
    age_std: float,
    source_map: Dict[str, int],
) -> SplitData:
    sample_ids = split_meta['sample_id'].tolist()
    x_expr = expr.loc[genes, sample_ids].T.values.astype(np.float32)
    x_expr = np.clip((x_expr - expr_mean) / expr_std, -8.0, 8.0)

    text_rows = [text_row_index[str(sample_id)] for sample_id in sample_ids]
    x_text = text_matrix[text_rows].astype(np.float32)

    age = split_meta['feat_age'].values.astype(float)
    age_mask = (~np.isnan(age)).astype(np.float32)
    age_filled = np.where(np.isnan(age), age_mean, age)
    age_target = ((age_filled - age_mean) / age_std).astype(np.float32)

    sex_info = np.array([sex_label(v) for v in split_meta['feat_sex']])
    sex_target = sex_info[:, 0].astype(np.int64)
    sex_mask = sex_info[:, 1].astype(np.float32)

    tumor_info = np.array([coarse_tumor(v) for v in split_meta['feat_tumor_status']])
    tumor_target = tumor_info[:, 0].astype(np.float32)
    tumor_mask = tumor_info[:, 1].astype(np.float32)

    source_target = np.array([source_map[safe_str(v)] for v in split_meta['gse']], dtype=np.int64)
    site_values = split_meta['feat_anatomical_site'].astype(str)
    site_counts = site_values.value_counts()
    site_keep = {k for k, v in site_counts.items() if v >= 25 and k not in {'unknown', 'nan', 'None'}}
    site_map = {label: idx for idx, label in enumerate(sorted(site_keep))}
    site_target = np.array([site_map.get(str(v), -1) for v in site_values], dtype=np.int64)

    disease_values = split_meta['feat_disease_label'].astype(str)
    disease_counts = disease_values.value_counts()
    disease_keep = {k for k, v in disease_counts.items() if v >= 25 and k not in {'unknown', 'nan', 'None'}}
    disease_map = {label: idx for idx, label in enumerate(sorted(disease_keep))}
    disease_target = np.array([disease_map.get(str(v), -1) for v in disease_values], dtype=np.int64)

    return SplitData(
        x_expr=x_expr,
        x_text=x_text,
        age_target=age_target,
        age_mask=age_mask,
        sex_target=sex_target,
        sex_mask=sex_mask,
        tumor_target=tumor_target,
        tumor_mask=tumor_mask,
        source_target=source_target,
        site_target=site_target,
        disease_target=disease_target,
        meta=split_meta.copy(),
    )


def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() < 1:
        return pred.sum() * 0.0
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1.0)


def masked_bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() < 1:
        return logit.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def masked_ce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() < 1:
        return logit.sum() * 0.0
    keep = mask > 0
    return F.cross_entropy(logit[keep], target[keep])


def recall_at_from_similarity(sim: np.ndarray, ks: List[int]) -> Dict[int, float]:
    if sim.shape[0] == 0:
        return {k: float('nan') for k in ks}
    max_k = min(max(ks), sim.shape[1])
    top_idx = np.argpartition(-sim, kth=max_k - 1, axis=1)[:, :max_k]
    row_ids = np.arange(sim.shape[0])[:, None]
    top_scores = sim[row_ids, top_idx]
    order = np.argsort(-top_scores, axis=1)
    ranked = top_idx[row_ids, order]
    labels = np.arange(sim.shape[0])
    out: Dict[int, float] = {}
    for k in ks:
        kk = min(k, ranked.shape[1])
        out[k] = float(np.mean([lab in ranked[i, :kk] for i, lab in enumerate(labels)]))
    return out


def get_batch_size(device: torch.device, eval_mode: bool = False) -> int:
    if device.type == 'cuda':
        return EVAL_BATCH_SIZE_GPU if eval_mode else BATCH_SIZE_GPU
    return EVAL_BATCH_SIZE_CPU if eval_mode else BATCH_SIZE_CPU


@dataclass
class TensorSplitData:
    x_expr: torch.Tensor
    x_text: torch.Tensor
    age_target: torch.Tensor
    age_mask: torch.Tensor
    sex_target: torch.Tensor
    sex_mask: torch.Tensor
    tumor_target: torch.Tensor
    tumor_mask: torch.Tensor
    source_target: torch.Tensor
    site_target: torch.Tensor
    disease_target: torch.Tensor
    sample_ids: np.ndarray


def to_tensor_split(split: SplitData) -> TensorSplitData:
    return TensorSplitData(
        x_expr=torch.tensor(split.x_expr, dtype=torch.float32),
        x_text=torch.tensor(split.x_text, dtype=torch.float32),
        age_target=torch.tensor(split.age_target, dtype=torch.float32),
        age_mask=torch.tensor(split.age_mask, dtype=torch.float32),
        sex_target=torch.tensor(split.sex_target, dtype=torch.long),
        sex_mask=torch.tensor(split.sex_mask, dtype=torch.float32),
        tumor_target=torch.tensor(split.tumor_target, dtype=torch.float32),
        tumor_mask=torch.tensor(split.tumor_mask, dtype=torch.float32),
        source_target=torch.tensor(split.source_target, dtype=torch.long),
        site_target=torch.tensor(split.site_target, dtype=torch.long),
        disease_target=torch.tensor(split.disease_target, dtype=torch.long),
        sample_ids=split.meta['sample_id'].values,
    )


def iterate_batches(split: TensorSplitData, batch_size: int, shuffle: bool):
    n = split.x_expr.shape[0]
    indices = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield (
            split.x_expr[idx],
            split.x_text[idx],
            split.age_target[idx],
            split.age_mask[idx],
            split.sex_target[idx],
            split.sex_mask[idx],
            split.tumor_target[idx],
            split.tumor_mask[idx],
            split.source_target[idx],
            split.site_target[idx],
            split.disease_target[idx],
            split.sample_ids[idx.cpu().numpy()],
        )


def multi_positive_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] <= 1:
        return logits.sum() * 0.0
    valid = labels >= 0
    if int(valid.sum()) <= 1:
        return logits.sum() * 0.0
    same = labels[:, None].eq(labels[None, :]) & valid[:, None] & valid[None, :]
    eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    same = same & ~eye
    if not bool(same.any()):
        return logits.sum() * 0.0
    logprob_row = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    row_mask = same.any(dim=1)
    row_pos = torch.logsumexp(logprob_row.masked_fill(~same, float('-inf')), dim=1)
    row_loss = -row_pos[row_mask].mean()

    logprob_col = logits.T - torch.logsumexp(logits.T, dim=1, keepdim=True)
    col_mask = same.any(dim=0)
    col_pos = torch.logsumexp(logprob_col.masked_fill(~same.T, float('-inf')), dim=1)
    col_loss = -col_pos[col_mask].mean()
    return 0.5 * (row_loss + col_loss)


def autocast_context(device: torch.device):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    return nullcontext()


def evaluate_loss(model: BulkRNALanguageAligner, split: TensorSplitData, device: torch.device, adv_lambda: float = 1.0) -> Dict[str, float]:
    model.eval()
    losses = []
    all_age_true, all_age_pred = [], []
    all_sex_true, all_sex_pred = [], []
    all_tumor_true, all_tumor_score = [], []
    expr_embs, text_embs = [], []
    batch_size = get_batch_size(device, eval_mode=True)
    with torch.no_grad():
        for batch in iterate_batches(split, batch_size=batch_size, shuffle=False):
            x_expr, x_text, age_t, age_m, sex_t, sex_m, tumor_t, tumor_m, src_t, site_t, disease_t, _ = batch
            x_expr = x_expr.to(device, non_blocking=True)
            x_text = x_text.to(device, non_blocking=True)
            age_t = age_t.to(device, non_blocking=True)
            age_m = age_m.to(device, non_blocking=True)
            sex_t = sex_t.to(device, non_blocking=True)
            sex_m = sex_m.to(device, non_blocking=True)
            tumor_t = tumor_t.to(device, non_blocking=True)
            tumor_m = tumor_m.to(device, non_blocking=True)
            src_t = src_t.to(device, non_blocking=True)
            site_t = site_t.to(device, non_blocking=True)
            disease_t = disease_t.to(device, non_blocking=True)

            with autocast_context(device):
                z_expr, z_text, logits, age_pred, sex_logit, tumor_logit, source_logit = model(x_expr, x_text, adv_lambda)
                loss = (
                    clip_loss(logits)
                    + SITE_POS_WEIGHT * multi_positive_loss(logits, site_t)
                    + DISEASE_POS_WEIGHT * multi_positive_loss(logits, disease_t)
                    + AUX_AGE_WEIGHT * masked_mse(age_pred, age_t, age_m)
                    + AUX_SEX_WEIGHT * masked_ce(sex_logit, sex_t, sex_m)
                    + AUX_TUMOR_WEIGHT * masked_bce(tumor_logit, tumor_t, tumor_m)
                    + ADV_WEIGHT * F.cross_entropy(source_logit, src_t)
                )
            losses.append(float(loss.item()))

            age_keep = age_m.cpu().numpy() > 0
            if age_keep.any():
                all_age_true.extend(age_t.cpu().numpy()[age_keep].tolist())
                all_age_pred.extend(age_pred.cpu().numpy()[age_keep].tolist())
            sex_keep = sex_m.cpu().numpy() > 0
            if sex_keep.any():
                all_sex_true.extend(sex_t.cpu().numpy()[sex_keep].tolist())
                all_sex_pred.extend(sex_logit.argmax(dim=1).cpu().numpy()[sex_keep].tolist())
            tumor_keep = tumor_m.cpu().numpy() > 0
            if tumor_keep.any():
                all_tumor_true.extend(tumor_t.cpu().numpy()[tumor_keep].tolist())
                all_tumor_score.extend(torch.sigmoid(tumor_logit).cpu().numpy()[tumor_keep].tolist())

            expr_embs.append(z_expr.cpu().numpy())
            text_embs.append(z_text.cpu().numpy())

    expr_mat = np.vstack(expr_embs)
    text_mat = np.vstack(text_embs)
    sim = expr_mat @ text_mat.T
    expr_recalls = recall_at_from_similarity(sim, [1, 5, 10])
    text_recalls = recall_at_from_similarity(sim.T, [1, 5, 10])

    metrics = {
        'loss': float(np.mean(losses)),
        'expr_to_text_r1': expr_recalls[1],
        'expr_to_text_r5': expr_recalls[5],
        'expr_to_text_r10': expr_recalls[10],
        'text_to_expr_r1': text_recalls[1],
        'text_to_expr_r5': text_recalls[5],
        'text_to_expr_r10': text_recalls[10],
    }
    if all_age_true:
        metrics['age_mae_z'] = float(mean_absolute_error(all_age_true, all_age_pred))
    if len(set(all_sex_true)) > 1:
        metrics['sex_acc'] = float(accuracy_score(all_sex_true, all_sex_pred))
    if len(set(all_tumor_true)) > 1:
        metrics['tumor_auroc'] = float(roc_auc_score(all_tumor_true, all_tumor_score))
    return metrics


def train_model(train: SplitData, val: SplitData, n_genes: int, n_text_features: int, n_sources: int) -> Tuple[BulkRNALanguageAligner, List[dict], int, float, float]:
    device = get_device()
    model = BulkRNALanguageAligner(n_genes, n_text_features, n_sources).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_tensor = to_tensor_split(train)
    val_tensor = to_tensor_split(val)

    best_state = None
    best_val = float('inf')
    best_epoch = -1
    bad_epochs = 0
    history: List[dict] = []
    start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        adv_lambda = min(1.0, epoch / 8)
        for batch in iterate_batches(train_tensor, batch_size=get_batch_size(device, eval_mode=False), shuffle=True):
            x_expr, x_text, age_t, age_m, sex_t, sex_m, tumor_t, tumor_m, src_t, site_t, disease_t, _ = batch
            x_expr = x_expr.to(device, non_blocking=True)
            x_text = x_text.to(device, non_blocking=True)
            age_t = age_t.to(device, non_blocking=True)
            age_m = age_m.to(device, non_blocking=True)
            sex_t = sex_t.to(device, non_blocking=True)
            sex_m = sex_m.to(device, non_blocking=True)
            tumor_t = tumor_t.to(device, non_blocking=True)
            tumor_m = tumor_m.to(device, non_blocking=True)
            src_t = src_t.to(device, non_blocking=True)
            site_t = site_t.to(device, non_blocking=True)
            disease_t = disease_t.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast_context(device):
                _, _, logits, age_pred, sex_logit, tumor_logit, source_logit = model(x_expr, x_text, adv_lambda)
                loss = (
                    clip_loss(logits)
                    + SITE_POS_WEIGHT * multi_positive_loss(logits, site_t)
                    + DISEASE_POS_WEIGHT * multi_positive_loss(logits, disease_t)
                    + AUX_AGE_WEIGHT * masked_mse(age_pred, age_t, age_m)
                    + AUX_SEX_WEIGHT * masked_ce(sex_logit, sex_t, sex_m)
                    + AUX_TUMOR_WEIGHT * masked_bce(tumor_logit, tumor_t, tumor_m)
                    + ADV_WEIGHT * F.cross_entropy(source_logit, src_t)
                    + 1e-4 * torch.sigmoid(model.gene_gate).mean()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))

        val_metrics = evaluate_loss(model, val_tensor, device, adv_lambda=1.0)
        row = {'epoch': epoch, 'train_loss': float(np.mean(batch_losses)), **val_metrics}
        history.append(row)
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.time() - start
    return model, history, best_epoch, best_val, train_seconds


def embed_split(model: BulkRNALanguageAligner, split: SplitData, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tensor_split = to_tensor_split(split)
    model.eval()
    expr_embs, text_embs = [], []
    age_pred, tumor_prob = [], []
    with torch.no_grad():
        for batch in iterate_batches(tensor_split, batch_size=get_batch_size(device, eval_mode=True), shuffle=False):
            x_expr, x_text, age_t, age_m, sex_t, sex_m, tumor_t, tumor_m, src_t, site_t, disease_t, sample_ids = batch
            x_expr = x_expr.to(device, non_blocking=True)
            x_text = x_text.to(device, non_blocking=True)
            with autocast_context(device):
                z_expr, z_text, logits, age_hat, sex_logit, tumor_logit, source_logit = model(x_expr, x_text, adv_lambda=1.0)
            expr_embs.append(z_expr.cpu().numpy())
            text_embs.append(z_text.cpu().numpy())
            age_pred.append(age_hat.cpu().numpy())
            tumor_prob.append(torch.sigmoid(tumor_logit).cpu().numpy())
    return np.vstack(expr_embs), np.vstack(text_embs), np.concatenate(age_pred), np.concatenate(tumor_prob)


def compute_query_results(
    model: BulkRNALanguageAligner,
    text_model: SentenceTransformer,
    all_meta: pd.DataFrame,
    all_expr_embeddings: np.ndarray,
    device: torch.device,
) -> Dict[str, List[dict]]:
    q_mat = encode_text_queries_backend(text_model, QUERY_LIBRARY)
    with torch.no_grad():
        q_emb = model.encode_text(torch.tensor(q_mat, dtype=torch.float32, device=device)).cpu().numpy()
    sims = q_emb @ all_expr_embeddings.T
    results: Dict[str, List[dict]] = {}
    keep_cols = ['sample_id', 'gse', 'feat_age', 'feat_sex', 'feat_anatomical_site', 'feat_tumor_status', 'feat_disease_label', 'caption_text']
    for i, query in enumerate(QUERY_LIBRARY):
        top_idx = np.argsort(-sims[i])[:10]
        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            rec = all_meta.iloc[idx][keep_cols].to_dict()
            rec['rank'] = rank
            rec['similarity'] = float(sims[i, idx])
            rows.append(rec)
        results[query] = rows
    return results


def encode_text_queries(model: BulkRNALanguageAligner, text_model: SentenceTransformer, queries: List[str], device: torch.device) -> np.ndarray:
    x = encode_text_queries_backend(text_model, queries)
    with torch.no_grad():
        emb = model.encode_text(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return emb


def compute_zero_shot_metrics(
    model: BulkRNALanguageAligner,
    text_model: SentenceTransformer,
    test_meta: pd.DataFrame,
    test_expr_emb: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    tumor_queries = [
        'a tumor transcriptome sample',
        'a non-tumor transcriptome sample',
    ]
    tumor_emb = encode_text_queries(model, text_model, tumor_queries, device)
    tumor_scores = test_expr_emb @ tumor_emb.T
    tumor_true = test_meta['feat_tumor_status'].astype(str).isin(['tumor', 'metastatic']).astype(int).values
    tumor_mask = test_meta['feat_tumor_status'].astype(str).isin(['tumor', 'metastatic', 'non_tumor', 'adjacent_normal', 'not_applicable']).values
    if tumor_mask.sum() > 5:
        tumor_prob = torch.softmax(torch.tensor(tumor_scores[tumor_mask], dtype=torch.float32), dim=1).numpy()[:, 0]
        metrics['zero_shot_tumor_auroc'] = float(roc_auc_score(tumor_true[tumor_mask], tumor_prob))

    sex_queries = [
        'a male transcriptome sample',
        'a female transcriptome sample',
    ]
    sex_emb = encode_text_queries(model, text_model, sex_queries, device)
    sex_scores = test_expr_emb @ sex_emb.T
    sex_true_map = {'male': 0, 'female': 1}
    sex_mask = test_meta['feat_sex'].astype(str).isin(sex_true_map).values
    if sex_mask.sum() > 5:
        sex_true = test_meta.loc[sex_mask, 'feat_sex'].map(sex_true_map).values
        sex_pred = sex_scores[sex_mask].argmax(axis=1)
        metrics['zero_shot_sex_acc'] = float(accuracy_score(sex_true, sex_pred))

    top_sites = test_meta['feat_anatomical_site'].astype(str).value_counts().head(6).index.tolist()
    site_queries = [f'a transcriptome sample from {site.replace("_", " ")} tissue' for site in top_sites]
    site_emb = encode_text_queries(model, text_model, site_queries, device)
    site_scores = test_expr_emb @ site_emb.T
    site_mask = test_meta['feat_anatomical_site'].astype(str).isin(top_sites).values
    if site_mask.sum() > 10:
        site_true_map = {s: i for i, s in enumerate(top_sites)}
        site_true = test_meta.loc[site_mask, 'feat_anatomical_site'].map(site_true_map).values
        site_pred = site_scores[site_mask].argmax(axis=1)
        metrics['zero_shot_site_acc_top6'] = float(accuracy_score(site_true, site_pred))

    age_queries = [
        'a young transcriptome sample',
        'a middle-aged transcriptome sample',
        'an old transcriptome sample',
    ]
    age_emb = encode_text_queries(model, text_model, age_queries, device)
    age_scores = test_expr_emb @ age_emb.T
    age_mask = test_meta['feat_age'].notna().values
    if age_mask.sum() > 10:
        age_series = test_meta.loc[age_mask, 'feat_age'].astype(float)
        bins = np.select(
            [age_series < 35, age_series < 60],
            [0, 1],
            default=2,
        )
        age_pred = age_scores[age_mask].argmax(axis=1)
        metrics['zero_shot_age3_acc'] = float(accuracy_score(bins, age_pred))

    return metrics


def make_gene_importance_table(model: BulkRNALanguageAligner, genes: List[str]) -> pd.DataFrame:
    gate = torch.sigmoid(model.gene_gate.detach().cpu()).numpy()
    first_layer = model.expr_encoder[0].weight.detach().cpu().numpy()
    weight_norm = np.linalg.norm(first_layer, axis=0)
    score = gate * weight_norm
    df = pd.DataFrame({'gene': genes, 'gate': gate, 'weight_norm': weight_norm, 'importance': score})
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj



def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    bundle = load_dataset()
    runtime_config = {
        'force_cpu_env': os.getenv('BMM_FORCE_CPU') == '1',
        'force_cuda_env': os.getenv('BMM_FORCE_CUDA') == '1',
        'cpu_threads': CPU_THREADS,
        'cpu_interop_threads': CPU_INTEROP_THREADS,
        'io_workers': IO_WORKERS,
        'loader_workers_cpu': LOADER_WORKERS_CPU,
        'loader_workers_gpu': LOADER_WORKERS_GPU,
        'prefetch_factor': PREFETCH_FACTOR,
        'batch_size_cpu': BATCH_SIZE_CPU,
        'eval_batch_size_cpu': EVAL_BATCH_SIZE_CPU,
        'batch_size_gpu': BATCH_SIZE_GPU,
        'eval_batch_size_gpu': EVAL_BATCH_SIZE_GPU,
        'host_mem_gb': round(get_total_memory_gb(), 2),
    }
    print(json.dumps({'runtime_config': runtime_config}, ensure_ascii=False))
    meta = bundle.meta.copy()
    meta['caption_text'] = meta.apply(build_caption, axis=1)
    train_meta, val_meta, test_meta = build_splits(meta)
    split_assignments = {**{sid: 'train' for sid in train_meta['sample_id']}, **{sid: 'val' for sid in val_meta['sample_id']}, **{sid: 'test' for sid in test_meta['sample_id']}}

    if bundle.layout == 'monolithic':
        genes = select_hvg(bundle.expr, train_meta['sample_id'].tolist())
    else:
        genes = select_hvg_sharded(bundle, train_meta['sample_id'].tolist())

    expr_train = load_expr_subset(bundle, genes, train_meta['sample_id'].tolist())
    x_train_raw = expr_train.T.values.astype(np.float32)
    _, _, expr_mean, expr_std = standardize_expr(x_train_raw, x_train_raw)

    text_model = load_text_model()
    text_meta = meta[['sample_id', 'caption_text']].drop_duplicates('sample_id').reset_index(drop=True)
    text_cache = OUTDIR / 'caption_text_embeddings.npy'
    text_matrix = encode_text_corpus(text_model, text_meta['caption_text'].tolist(), text_cache)
    text_row_index = {str(sample_id): idx for idx, sample_id in enumerate(text_meta['sample_id'].tolist())}

    age_train = train_meta['feat_age'].values.astype(float)
    age_mean = float(np.nanmean(age_train))
    age_std = float(np.nanstd(age_train))
    if age_std < 1e-6:
        age_std = 1.0

    sources = sorted(meta['gse'].astype(str).unique())
    source_map = {s: i for i, s in enumerate(sources)}

    # Use unified standardization parameters for split preparation
    expr_val = load_expr_subset(bundle, genes, val_meta['sample_id'].tolist())
    expr_test = load_expr_subset(bundle, genes, test_meta['sample_id'].tolist())
    train_split = prepare_split_data(expr_train, train_meta, genes, text_matrix, text_row_index, expr_mean, expr_std, age_mean, age_std, source_map)
    val_split = prepare_split_data(expr_val, val_meta, genes, text_matrix, text_row_index, expr_mean, expr_std, age_mean, age_std, source_map)
    test_split = prepare_split_data(expr_test, test_meta, genes, text_matrix, text_row_index, expr_mean, expr_std, age_mean, age_std, source_map)

    model, history, best_epoch, best_val_loss, train_seconds = train_model(
        train_split,
        val_split,
        n_genes=len(genes),
        n_text_features=int(text_matrix.shape[1]),
        n_sources=len(source_map),
    )

    device = get_device()
    model = model.to(device)
    test_metrics = evaluate_loss(model, to_tensor_split(test_split), device, adv_lambda=1.0)
    test_expr_emb, test_text_emb, _, _ = embed_split(model, test_split, device)
    zero_shot_metrics = compute_zero_shot_metrics(model, text_model, test_meta, test_expr_emb, device)

    all_meta = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)
    expr_all = load_expr_subset(bundle, genes, all_meta['sample_id'].tolist())
    all_split = prepare_split_data(expr_all, all_meta, genes, text_matrix, text_row_index, expr_mean, expr_std, age_mean, age_std, source_map)
    expr_emb, text_emb, age_pred_z, tumor_prob = embed_split(model, all_split, device)
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(expr_emb)

    age_pred = age_pred_z * age_std + age_mean
    expr_text_gap = 1.0 - np.sum(expr_emb * text_emb, axis=1)
    hover = []
    for row, pred_age, tprob, gap, (x, y) in zip(all_meta.itertuples(), age_pred, tumor_prob, expr_text_gap, coords):
        hover.append(
            f"sample={row.sample_id}<br>gse={row.gse}<br>age={row.feat_age}<br>pred_age={pred_age:.1f}<br>sex={row.feat_sex}<br>tumor={row.feat_tumor_status}<br>site={row.feat_anatomical_site}<br>disease={row.feat_disease_label}<br>tumor_prob={tprob:.3f}<br>expr_text_gap={gap:.3f}"
        )

    sample_embeddings = all_meta[['sample_id', 'gse', 'feat_age', 'feat_sex', 'feat_anatomical_site', 'feat_tumor_status', 'feat_disease_label', 'caption_text']].copy()
    sample_embeddings['split'] = sample_embeddings['sample_id'].map(split_assignments)
    sample_embeddings['pca1'] = coords[:, 0]
    sample_embeddings['pca2'] = coords[:, 1]
    sample_embeddings['pred_age'] = age_pred
    sample_embeddings['tumor_probability'] = tumor_prob
    sample_embeddings['expr_text_gap'] = expr_text_gap
    sample_embeddings['sample_hover'] = hover

    query_results = compute_query_results(model, text_model, all_meta.reset_index(drop=True), expr_emb, device)
    top_genes = make_gene_importance_table(model, genes)
    caption_examples = pd.concat([
        train_meta[['sample_id', 'gse', 'feat_tumor_status', 'caption_text']].head(5).assign(split='train'),
        val_meta[['sample_id', 'gse', 'feat_tumor_status', 'caption_text']].head(5).assign(split='val'),
        test_meta[['sample_id', 'gse', 'feat_tumor_status', 'caption_text']].head(8).assign(split='test'),
    ], ignore_index=True)

    torch.save({
        'state_dict': model.state_dict(),
        'selected_genes': genes,
        'text_backend': 'sentence_transformer',
        'text_encoder_name': TEXT_MODEL_NAME,
        'text_encoder_revision': TEXT_MODEL_REVISION,
        'expr_mean': expr_mean.tolist(),
        'expr_std': expr_std.tolist(),
        'age_mean': age_mean,
        'age_std': age_std,
        'source_map': source_map,
        'split_assignments': split_assignments,
        'config': {
            'embed_dim': EMBED_DIM,
            'n_features': len(genes),
            'n_text_features': int(text_matrix.shape[1]),
            'text_backend': 'sentence_transformer',
            'text_encoder_name': TEXT_MODEL_NAME,
            'text_encoder_revision': TEXT_MODEL_REVISION,
            'seed': SEED,
            'split_manifest': SPLIT_MANIFEST or None,
        }
    }, OUTDIR / 'bulk_multimodal_embedding.pt')

    pd.DataFrame(history).to_csv(OUTDIR / 'training_history.csv', index=False)
    sample_embeddings.to_csv(OUTDIR / 'sample_embeddings.csv', index=False)
    top_genes.to_csv(OUTDIR / 'top_embedding_genes.csv', index=False)
    (OUTDIR / 'query_results.json').write_text(json.dumps(to_native(query_results), ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {
        'n_samples': int(len(meta)),
        'n_gse': int(meta['gse'].nunique()),
        'n_features': int(len(genes)),
        'n_text_features': int(text_matrix.shape[1]),
        'text_backend': 'sentence_transformer',
        'text_encoder_name': TEXT_MODEL_NAME,
        'text_encoder_revision': TEXT_MODEL_REVISION,
        'embed_dim': EMBED_DIM,
        'seed': SEED,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'train_seconds': float(train_seconds),
        'device': str(device),
        'data_dir': str(DATA_DIR),
        'data_layout': bundle.layout,
        'run_name': RUN_NAME,
        'batch_size_train': get_batch_size(device, eval_mode=False),
        'batch_size_eval': get_batch_size(device, eval_mode=True),
        'io_workers': IO_WORKERS,
        'loader_workers': LOADER_WORKERS_GPU if device.type == 'cuda' else LOADER_WORKERS_CPU,
        'prefetch_factor': PREFETCH_FACTOR,
        'cpu_threads': CPU_THREADS,
        'cpu_interop_threads': CPU_INTEROP_THREADS,
        'host_mem_gb': round(get_total_memory_gb(), 2),
        'split_manifest': SPLIT_MANIFEST or None,
        'device_probe': to_native(DEVICE_PROBE or {}),
        'test_metrics': to_native(test_metrics),
        'zero_shot_metrics': to_native(zero_shot_metrics),
        'split_counts': {k: int(v) for k, v in all_meta['sample_id'].map(split_assignments).value_counts().to_dict().items()},
        'source_dataset_counts': {k: int(v) for k, v in meta.get('source_dataset', pd.Series(dtype=str)).value_counts().to_dict().items()},
        'method_reference': {
            'original_transcriptome_tower': 'Frozen Geneformer on ranked genes',
            'this_project_transcriptome_tower': 'Bulk log-expression MLP encoder with gene gate',
            'original_text_tower': 'BioBERT',
            'this_project_text_tower': 'SentenceTransformer embeddings + MLP text encoder',
            'shared_training_objective': 'Symmetric InfoNCE / CLIP-style transcriptome-text alignment',
            'project_specific_extension': 'GSE source-adversarial head plus age/sex/tumor auxiliary heads',
        },
        'sources': {
            'reference_paper': 'https://www.nature.com/articles/s41587-025-02857-9',
            'reference_scope': 'transcriptome-language alignment literature',
        }
    }
    (OUTDIR / 'summary.json').write_text(json.dumps(to_native(summary), ensure_ascii=False, indent=2), encoding='utf-8')

    artifacts = TrainArtifacts(
        model=model,
        text_backend='sentence_transformer',
        text_encoder_name=TEXT_MODEL_NAME,
        selected_genes=genes,
        expr_mean=expr_mean,
        expr_std=expr_std,
        age_mean=age_mean,
        age_std=age_std,
        source_map=source_map,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_seconds=train_seconds,
        split_assignments=split_assignments,
    )
    print(json.dumps(to_native(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
