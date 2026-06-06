"""Data loading: CSV → PyTorch Dataset → DataLoader."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

from src import config as cfg


class SlabThicknessDataset(Dataset):
    """Dataset for ionospheric slab-thickness prediction."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_data() -> tuple[DataLoader, DataLoader, StandardScaler]:
    """Load CSV, preprocess, return (train_loader, val_loader, target_scaler)."""
    df = pd.read_csv(cfg.DATA_PATH)

    # ── optional subset filter ──
    if cfg.SUBSET_FILTER and cfg.SUBSET_FILTER in df.columns:
        df = df[df["subset"] == cfg.SUBSET_FILTER]
        print(f"[data] subset='{cfg.SUBSET_FILTER}' → {len(df):,} rows")

    # ── optional sampling for quick validation ──
    if cfg.SAMPLE_LIMIT and len(df) > cfg.SAMPLE_LIMIT:
        df = df.sample(n=cfg.SAMPLE_LIMIT, random_state=42)
        print(f"[data] sampled {cfg.SAMPLE_LIMIT:,} rows")

    # ── feature columns (ordered) ──
    feature_cols = []
    for g in cfg.FEATURE_GROUPS.values():
        feature_cols.extend(g)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from CSV: {missing}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[cfg.TARGET_COL].values.astype(np.float32)

    # ── target scaling (StandardScaler) ──
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # ── train / val split ──
    full_dataset = SlabThicknessDataset(X, y)
    val_size = int(len(full_dataset) * cfg.VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"[data] train={train_size:,}  val={val_size:,}  features={len(feature_cols)}")
    return train_loader, val_loader, scaler


def build_feature_index() -> dict[str, list[int]]:
    """Return {group_name: [column_indices]} for the model's branch routing."""
    indices: dict[str, list[int]] = {}
    offset = 0
    for group, cols in cfg.FEATURE_GROUPS.items():
        indices[group] = list(range(offset, offset + len(cols)))
        offset += len(cols)
    return indices
