"""Training loop with validation, early-stopping, and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from src import config as cfg
from src.data_loader import load_data
from src.model import SlabThicknessNet


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def train() -> tuple[SlabThicknessNet, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    train_loader, val_loader, scaler = load_data()

    model = SlabThicknessNet().to(device)
    print(f"[train] params = {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.SmoothL1Loss()
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                       weight_decay=cfg.WEIGHT_DECAY)

    if cfg.SCHEDULER == "cosine":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.T_0, T_mult=cfg.T_MULT, eta_min=cfg.LR_MIN,
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = val_epoch(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - t0

        if cfg.SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"epoch {epoch:3d}/{cfg.EPOCHS}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={lr:.2e}  time={elapsed:.1f}s")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ new best")

        if epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
            print(f"[train] early stop at epoch {epoch}")
            break

    # restore best
    model.load_state_dict(best_state)
    torch.save(best_state, cfg.CHECKPOINT_DIR / f"best_model{cfg.CKPT_SUFFIX}.pt")

    # save scaler metadata
    torch.save({"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
               cfg.CHECKPOINT_DIR / f"scaler{cfg.CKPT_SUFFIX}.pt")

    print(f"[train] best epoch={best_epoch}  val_loss={best_loss:.6f}")
    return model, history
