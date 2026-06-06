# Ionospheric Slab Thickness Prediction — Residual MLP

PyTorch implementation of a multi-branch Residual MLP for predicting ionospheric equivalent slab thickness (τ, in km) from GNSS radio occultation data.

## Quick Start

```bash
# train on full dataset
python run.py

# evaluate on test set (uses checkpoints/best_model.pt)
python -c "from src.evaluate import evaluate; evaluate()"
```

Requires `PyTorch >= 2.0`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`.  
Place `model_input_with_raw.csv` under `data/`.

## Project Structure

```
├── run.py               # Entry point: train → plot → evaluate
├── src/
│   ├── config.py        # Hyperparameters, feature groups, paths
│   ├── data_loader.py   # CSV → PyTorch Dataset → DataLoader
│   ├── model.py         # SlabThicknessNet (multi-branch Residual MLP)
│   ├── train.py         # Training loop + early stopping + checkpoint
│   └── evaluate.py      # Test-set metrics + plots + persistent results
├── data/
│   ├── dataset.txt      # Feature descriptions
│   └── model_input_with_raw.csv   # Dataset (git-ignored, ~685 MB)
├── doc/
│   ├── resnet.md        # Network design rationale
│   └── vae.md
├── result/              # Persisted evaluation outputs
└── checkpoints/         # Saved model weights & scaler (git-ignored)
```

## Model Architecture

```
Input (12) ─┬─ Physical branch spatial(4)+solar(3) ─→ [128→256] ─┐
            │                                                       ├─ Fusion 384
            └─ Temporal branch temporal(5) ────────────→ [64→128] ─┘
                                                               │
                                    4 × Residual Block ─→ Head ─→ τ (km)
```

Each Residual Block: `Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm` with skip connection. See `doc/resnet.md` for full design rationale.

## Features (12) → Target (τ)

| Group | Features | Description |
|-------|----------|-------------|
| Spatial (4) | sin/cos(lon), lat, mlat | Geographic + geomagnetic position |
| Solar (3) | Kp, F10.7, vTEC | Solar & ionospheric activity |
| Temporal (5) | sin/cos(LT), sin/cos(DOY), cos(SZA) | Diurnal, seasonal cycles |

Target: `tau_km` — equivalent slab thickness, ~100–700 km.

## Results

| Metric | Residual MLP | XGBoost+EL (paper) | Δ |
|--------|:---:|:---:|:---:|
| RMSE | **59.6 km** | 62.3 km | −2.7 |
| MAE  | 43.4 km | 41.5 km | +1.9 |
| MAPE | 12.5% | 13.1% | −0.6 |
| R    | 0.841 | 0.904 | −0.06 |

## Training Config

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T0=20) |
| Loss | SmoothL1Loss (β=1.0) |
| Batch size | 4096 |
| Early stop | patience=15 |
| Params | ~1.88M |