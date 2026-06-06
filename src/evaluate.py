"""Test-set evaluation: metrics + visualisation + persistent results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config as cfg
from src.data_loader import SlabThicknessDataset
from src.model import SlabThicknessNet

@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, truths = [], []
    for x, y in loader:
        x = x.to(device)
        p = model(x).cpu().numpy()
        preds.append(p)
        truths.append(y.numpy())
    return np.concatenate(preds).ravel(), np.concatenate(truths).ravel()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff / (y_true + 1e-8))) * 100)
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "r": r}


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    """Scatter + residual distribution."""
    diff = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. pred vs true scatter
    ax = axes[0]
    ax.hexbin(y_true, y_pred, gridsize=80, cmap="inferno", mincnt=1)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", alpha=0.6, label="perfect")
    ax.set_xlabel("True τ (km)")
    ax.set_ylabel("Predicted τ (km)")
    ax.set_title("Prediction vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. residual vs true
    ax = axes[1]
    ax.hexbin(y_true, diff, gridsize=80, cmap="inferno", mincnt=1)
    ax.axhline(0, color="r", linestyle="--", alpha=0.6)
    ax.set_xlabel("True τ (km)")
    ax.set_ylabel("Residual (km)")
    ax.set_title("Residual vs Ground Truth")
    ax.grid(True, alpha=0.3)

    # 3. residual histogram
    ax = axes[2]
    ax.hist(diff, bins=100, density=True, alpha=0.7, color="steelblue",
            edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="r", linestyle="--", alpha=0.6)
    ax.axvline(np.mean(diff), color="orange", linestyle="-", alpha=0.8,
               label=f"mean = {diff.mean():.2f} km")
    ax.set_xlabel("Residual (km)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / "evaluation_plots.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[plot] saved {save_path}")


def save_results(y_true: np.ndarray, y_pred: np.ndarray,
                 m: dict[str, float], out_dir: Path, ckpt_suffix: str):
    """Persist predictions, metrics, and summary to result/."""

    # 1. predictions (first 10k samples + random 10k)
    n_sample = min(10000, len(y_true))
    idx = np.random.default_rng(42).choice(len(y_true), size=n_sample, replace=False)
    pred_df = pd.DataFrame({
        "y_true_km": np.round(y_true[idx], 3),
        "y_pred_km": np.round(y_pred[idx], 3),
        "residual_km": np.round(y_pred[idx] - y_true[idx], 3),
    })
    csv_path = out_dir / f"predictions{ckpt_suffix}.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    # 2. metrics JSON
    json_path = out_dir / f"metrics{ckpt_suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"[save] {json_path}")

    # 3. summary text
    txt_path = out_dir / f"summary{ckpt_suffix}.txt"
    bl_rmse, bl_mae, bl_mape, bl_r = 62.3, 41.5, 13.1, 0.904
    lines = [
        "=" * 60,
        "  Ionospheric Slab Thickness Prediction — Test Results",
        "=" * 60,
        "",
        f"  Model     : SlabThicknessNet (Residual MLP, ~1.88M params)",
        f"  Checkpoint: best_model{ckpt_suffix}.pt / scaler{ckpt_suffix}.pt",
        f"  Test size : {len(y_true):,} samples",
        "",
        "─── Metrics ───",
        f"  RMSE : {m['rmse']:.2f} km",
        f"  MAE  : {m['mae']:.2f} km",
        f"  MAPE : {m['mape']:.2f} %",
        f"  R²   : {m['r2']:.4f}",
        f"  R    : {m['r']:.4f}",
        "",
        "─── vs. XGBoost+EL (paper baseline) ───",
        f"  RMSE : {m['rmse']:.1f}  (baseline {bl_rmse})  Δ = {m['rmse'] - bl_rmse:+.1f} km",
        f"  MAE  : {m['mae']:.1f}  (baseline {bl_mae})  Δ = {m['mae'] - bl_mae:+.1f} km",
        f"  MAPE : {m['mape']:.1f}%  (baseline {bl_mape}%)  Δ = {m['mape'] - bl_mape:+.1f}%",
        f"  R    : {m['r']:.3f}  (baseline {bl_r})  Δ = {m['r'] - bl_r:+.3f}",
        "",
        f"  Residual mean : {np.mean(y_pred - y_true):.3f} km",
        f"  Residual std  : {np.std(y_pred - y_true):.3f} km",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[save] {txt_path}")


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")

    # ── load scaler ──
    scaler_path = cfg.CHECKPOINT_DIR / f"scaler{cfg.CKPT_SUFFIX}.pt"
    model_path = cfg.CHECKPOINT_DIR / f"best_model{cfg.CKPT_SUFFIX}.pt"
    print(f"[eval] scaler: {scaler_path}")
    print(f"[eval] model : {model_path}")

    meta = torch.load(scaler_path, map_location="cpu", weights_only=False)
    scaler_mean = meta["scaler_mean"]
    scaler_scale = meta["scaler_scale"]

    # ── load model ──
    model = SlabThicknessNet().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # ── load test set ──
    df = pd.read_csv(cfg.DATA_PATH)
    print(f"[eval] total rows in CSV: {len(df):,}")

    if "subset" in df.columns:
        subsets = df["subset"].value_counts().to_dict()
        print(f"[eval] subset distribution: {subsets}")
        df = df[df["subset"] == "test"]
        print(f"[eval] test rows = {len(df):,}")

    if len(df) == 0:
        print("[eval] ERROR: no test rows found. Checking for 'val' subset...")
        df_all = pd.read_csv(cfg.DATA_PATH)
        if "subset" in df_all.columns:
            df = df_all[df_all["subset"] == "val"]
            print(f"[eval] using val subset: {len(df):,} rows")
        if len(df) == 0:
            print("[eval] no val subset either, sampling from full dataset")
            df = df_all.sample(n=min(100000, len(df_all)), random_state=42)

    feature_cols = []
    for g in cfg.FEATURE_GROUPS.values():
        feature_cols.extend(g)

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[cfg.TARGET_COL].values.astype(np.float32)

    y = (y_raw.reshape(-1, 1) - scaler_mean) / scaler_scale
    y = y.ravel().astype(np.float32)

    ds = SlabThicknessDataset(X, y)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
                         num_workers=2, pin_memory=True)

    y_pred_scaled, y_true_scaled = predict(model, loader, device)

    y_pred_km = y_pred_scaled * scaler_scale[0] + scaler_mean[0]
    y_true_km = y_true_scaled * scaler_scale[0] + scaler_mean[0]

    m = metrics(y_true_km, y_pred_km)
    print("\n─── test metrics ───")
    print(f"  RMSE : {m['rmse']:.2f} km")
    print(f"  MAE  : {m['mae']:.2f} km")
    print(f"  MAPE : {m['mape']:.2f} %")
    print(f"  R²   : {m['r2']:.4f}")
    print(f"  R    : {m['r']:.4f}")

    print("\n─── vs. XGBoost+EL (paper baseline) ───")
    bl_rmse, bl_mae, bl_mape, bl_r = 62.3, 41.5, 13.1, 0.904
    print(f"  RMSE : {m['rmse']:.1f}  (baseline {bl_rmse})  Δ = {m['rmse'] - bl_rmse:+.1f} km")
    print(f"  MAE  : {m['mae']:.1f}  (baseline {bl_mae})  Δ = {m['mae'] - bl_mae:+.1f} km")
    print(f"  MAPE : {m['mape']:.1f}%  (baseline {bl_mape}%)  Δ = {m['mape'] - bl_mape:+.1f}%")
    print(f"  R    : {m['r']:.3f}  (baseline {bl_r})  Δ = {m['r'] - bl_r:+.3f}")

    # ── persist results ──
    plot_results(y_true_km, y_pred_km, cfg.RESULT_DIR)
    save_results(y_true_km, y_pred_km, m, cfg.RESULT_DIR, cfg.CKPT_SUFFIX)
    return m


if __name__ == "__main__":
    evaluate()
