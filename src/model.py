"""Multi-branch Residual MLP for ionospheric slab-thickness prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as cfg


class ResidualBlock(nn.Module):
    """Linear → LayerNorm → GELU → Dropout → Linear → LayerNorm, with skip."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = cfg.DROPOUT):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.ln2 = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = out + residual
        return F.gelu(out)


class FeatureBranch(nn.Module):
    """A small FC tower for a subset of input features."""

    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float = cfg.DROPOUT):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h, bias=False),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SlabThicknessNet(nn.Module):
    """Multi-branch Residual MLP for slab-thickness regression.

    Physical branch (spatial + solar) and temporal branch extract features
    independently, then fuse into a shared residual trunk.
    """

    def __init__(self):
        super().__init__()

        # ── branches ──
        self.phys_branch = FeatureBranch(
            in_dim=cfg.SPATIAL_DIM + cfg.SOLAR_DIM,   # 4 + 3 = 7
            hidden_dims=cfg.BRANCH_HIDDEN,              # [128, 256]
        )
        self.temp_branch = FeatureBranch(
            in_dim=cfg.TEMPORAL_DIM,                    # 5
            hidden_dims=[64, 128],
        )

        # ── fusion projection ──
        fusion_in = cfg.BRANCH_HIDDEN[-1] + 128          # 256 + 128 = 384
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.FUSION_DIM, bias=False),
            nn.LayerNorm(cfg.FUSION_DIM),
            nn.GELU(),
        )

        # ── residual trunk ──
        self.res_blocks = nn.ModuleList()
        for in_d, h_d in cfg.RESIDUAL_BLOCKS:
            self.res_blocks.append(ResidualBlock(in_d, h_d, h_d))

        # ── prediction head ──
        head_in = cfg.RESIDUAL_BLOCKS[-1][1]  # 128
        head_layers = []
        prev = head_in
        for h in cfg.HEAD_DIMS[:-1]:
            head_layers.extend([
                nn.Linear(prev, h, bias=False),
                nn.LayerNorm(h),
                nn.GELU(),
            ])
            prev = h
        head_layers.append(nn.Linear(prev, cfg.HEAD_DIMS[-1]))  # → 1
        self.head = nn.Sequential(*head_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # route features to branches
        phys_end = cfg.SPATIAL_DIM + cfg.SOLAR_DIM
        phys_feat = x[:, :phys_end]          # first 7 columns
        temp_feat = x[:, phys_end:]           # remaining 5 columns

        phys_out = self.phys_branch(phys_feat)
        temp_out = self.temp_branch(temp_feat)

        x = torch.cat([phys_out, temp_out], dim=1)
        x = self.fusion(x)

        for block in self.res_blocks:
            x = block(x)

        return self.head(x)
