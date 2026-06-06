# 电离层等效板厚预测 — 变分自编码器 (VAE) 数据增强设计

## 一、设计动机

现有 `SlabThicknessNet` 是一个确定性回归模型：输入 12 维特征 → 输出 τ。训练完全依赖真实样本，在以下方面存在提升空间：

1. **数据覆盖不均**：训练集中某些 τ 区间的样本稀疏（如极高/极低板厚），导致模型在这些区域泛化差
2. **无法利用逆映射**：给定 τ，哪些 (特征, τ) 组合是物理上合理的？这个信息未被利用
3. **一对多关系未建模**：同一个 τ 可以对应多种特征组合（如高纬强太阳活动 ≈ 低纬弱太阳活动的板厚可能相近）

**核心思路**：训练一个编码器学习条件分布 P(features | τ)，从 τ 采样出合理的特征组合，作为解码器的**数据增强源**。解码器在真实数据（全权重）+ 采样数据（降权重）上联合训练，利用合成样本填补真实数据的覆盖空隙，提升最终预测精度。

**编码器的定位**：它是解码器的"数据增强器"——训练时为解码器提供更多样的 (特征, τ) 配对，推理时不需要它。

---

## 二、架构总览

```
   ┌──────────────────────────────────────────────────┐
   │                  训练阶段                          │
   │                                                   │
   │   τ_true ──→ Encoder ──→ μ, log σ²               │
   │                  │          │                     │
   │                  │    z = μ + ε·σ  (采样)         │
   │                  │          │                     │
   │                  │    ┌─────▼──────┐              │
   │                  │    │  Decoder   │ ←── x_true  │
   │                  │    │ (主模型)   │              │
   │                  │    └──┬───┬────┘              │
   │                  │       │   │                    │
   │                  │  τ̂_aug   τ̂_sup               │
   │                  │   │       │                    │
   │                  │   ▼       ▼                    │
   │                  │ L_aug   L_sup                  │
   │                  │ (α权重) (全权重)                │
   │                  │                               │
   │   x_true ────────┼──→ L_enc ←── μ (特征重建)     │
   │                  │                               │
   │                  └──→ L_KL  ←── μ, log σ²        │
   │                                                   │
   └──────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────┐
   │                  推理阶段                          │
   │                                                   │
   │   x_true ──→ Decoder ──→ τ̂                       │
   │   (编码器不参与推理)                               │
   └──────────────────────────────────────────────────┘
```

**数据流（每个 batch）**：

| 路径 | 数据流 | 损失 | 权重 | 作用 |
|------|--------|------|------|------|
| 监督路径 | x_true → Decoder → τ̂_sup | SmoothL1(τ̂_sup, τ_true) | 1.0 | 主训练信号 |
| 增强路径 | τ_true → Encoder → z ~ N(μ, σ²) → Decoder → τ̂_aug | SmoothL1(τ̂_aug, τ_true) | α (≈0.3) | 数据增强 |
| 编码器监督 | τ_true → Encoder → μ | SmoothL1(μ, x_true) | γ (≈0.5) | 确保编码器学会预测特征 |
| KL 正则化 | Encoder 输出 (μ, log σ²) | KL(N(μ,σ²) ‖ N(0,I)) | β (≈0.001) | 防止方差坍缩 |

---

## 三、编码器设计（Encoder）—— 特征生成器

### 3.1 输入

| 项目 | 说明 |
|------|------|
| 原始目标 | `tau_km`，物理单位 km，范围 ~100–700 |
| 训练目标 | 经 `StandardScaler` 变换为 N(0, 1) |
| 输入维度 | 1 |
| 输入预处理 | 复用与 SlabThicknessNet 相同的 StandardScaler |

### 3.2 架构

```
   tau_km (1)
       │
  Linear(1 → 64) + LayerNorm + GELU + Dropout
       │
  Linear(64 → 128) + LayerNorm + GELU + Dropout
       │
  Linear(128 → 256) + LayerNorm + GELU + Dropout
       │
       ├──────────────────┤
       ▼                  ▼
  Linear(256 → 12)   Linear(256 → 12)
     μ (均值)          log σ² (对数方差)
       │                  │
       └────────┬─────────┘
                ▼
     z = μ + ε · exp(0.5 · log σ²)    ε ~ N(0, I)
     (重参数化技巧，12 维)
                │
                ▼
         送入 Decoder（增强路径）
         或与 x_true 计算 L_enc（μ 的监督损失）
```

### 3.3 输出

| 输出头 | 维度 | 激活函数 | 物理含义 |
|--------|------|----------|----------|
| μ | 12 | 无（线性） | 给定 τ，各特征的期望值 |
| log σ² | 12 | 无（线性） | 给定 τ，各特征的预测不确定性 |

12 维输出对应全部 12 个输入特征（与解码器输入列顺序严格一致）：

| 索引 | 特征 | 分组 |
|------|------|------|
| 0–3 | `proc_sin_lon`, `proc_cos_lon`, `proc_lat`, `proc_mlat` | 空间 |
| 4–6 | `proc_kp`, `proc_f107`, `proc_vtec` | 太阳/电离层 |
| 7–11 | `proc_sin_lt`, `proc_cos_lt`, `proc_sin_doy`, `proc_cos_doy`, `proc_cos_chi` | 时间 |

### 3.4 设计选择

| 选择 | 理由 |
|------|------|
| 隐藏维度 64→128→256 | 从 1 维标量扩展到 256 维隐藏表示，提供足够容量学习 τ → 12 维特征的复杂逆映射 |
| 线性输出（无激活函数） | μ 需覆盖各特征的值域（[-1,1] 和 [0,1]）；log σ² 值域为全体实数。KL + 监督损失自然约束其范围 |
| 单塔 MLP，不设分支 | 输入只有 1 维标量，无分组特征需要分开处理 |
| LayerNorm + GELU + Dropout | 与现有模型组件一致 |
| 参数量 | 约 62K |

---

## 四、解码器设计（Decoder）—— 预测模型

### 4.1 复用 SlabThicknessNet

解码器**完全复用** `SlabThicknessNet`（`src/model.py`），不做任何结构修改：

| 项目 | 说明 |
|------|------|
| 架构 | 双分支残差 MLP（物理分支 + 时间分支 → 融合 → 残差主干 → 预测头） |
| 输入 | 12 维特征向量 |
| 输出 | 1 维 τ̂（StandardScaler 空间，无激活函数） |
| 参数量 | ~1.88M |

### 4.2 两种输入模式

| 模式 | 输入来源 | 损失权重 | 使用场景 |
|------|----------|----------|----------|
| 监督模式 | 真实特征 x_true | 1.0（全权重） | 每个 batch 都计算 |
| 增强模式 | 编码器采样 z ~ N(μ, σ²) | α ≈ 0.3（降权重） | 每个 batch 都计算 |

解码器对两种输入执行相同的计算图，仅损失权重不同。这意味着编码器采样出"足够真实"的特征时，解码器能从中受益；即使采样质量不佳，降权重也限制了负面影响。

### 4.3 推理

推理时仅使用解码器 + 真实特征，与原始 SlabThicknessNet 完全一致：

```
x_true → Decoder → τ̂ → inverse StandardScaler → τ̂_km
```

---

## 五、损失函数设计

### 5.1 总损失

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \alpha \cdot \mathcal{L}_{\text{aug}} + \gamma \cdot \mathcal{L}_{\text{enc}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

四项损失各司其职：

| 损失项 | 公式 | 权重 | 作用 |
|--------|------|------|------|
| L_sup | SmoothL1(Decoder(x_true), τ_true) | 1.0 | 主任务：真实特征 → τ |
| L_aug | SmoothL1(Decoder(z), τ_true) | α ≈ 0.3 | 增强任务：采样特征 → τ（降权） |
| L_enc | SmoothL1(μ, x_true) | γ ≈ 0.5 | 编码器监督：μ 必须能预测真实特征 |
| L_KL | KL(N(μ,σ²) ‖ N(0,I)) | β ≈ 0.001 | 正则化：防止 log σ² → -∞ |

### 5.2 监督损失 L_sup（主训练信号）

$$\mathcal{L}_{\text{sup}} = \text{SmoothL1Loss}(\text{Decoder}(x_{\text{true}}), \tau_{\text{true}})$$

与原始 SlabThicknessNet 完全相同的训练目标。权重始终为 1.0，是训练的主导信号。

### 5.3 增强损失 L_aug（数据增强信号）

$$\mathcal{L}_{\text{aug}} = \text{SmoothL1Loss}(\text{Decoder}(z), \tau_{\text{true}}), \quad z \sim N(\mu, \sigma^2)$$

| 项目 | 说明 |
|------|------|
| 作用 | 让解码器从"非真实但物理合理"的特征中学习 τ 的映射 |
| 权重 α | 0.3，降低合成数据对训练的干扰 |
| 梯度流向 | 通过 Decoder → z（重参数化）→ Encoder。Encoder 也会收到"产生让 τ 预测更准的特征"的信号 |

**α 的选择逻辑**：

| α 值 | 效果 |
|------|------|
| 0.1 | 增强信号很弱，接近不使用增强 |
| 0.3 | 平衡点：编码器有一定影响但不过度干预主训练 |
| 0.5 | 合成数据与真实数据等权重，可能引入噪声 |
| 1.0 | 等同真实数据，编码器误差会被放大 |

### 5.4 特征重建损失 L_enc（编码器监督）

$$\mathcal{L}_{\text{enc}} = \text{SmoothL1Loss}(\mu, x_{\text{true}})$$

这是本设计与标准 VAE 的**关键区别**。如果不加这个损失，编码器仅通过以下路径学习：

- L_aug 的梯度通过 Decoder → z → Encoder：这会驱动编码器产生"让解码器任务更容易"的特征，而非"真实"的特征。极端情况下，编码器可能将所有 τ 映射到同一个 z，解码器退化为常数函数
- L_KL 的梯度：仅约束分布形状，不保证 μ 与真实特征的语义关联

L_enc 直接监督 μ，强制编码器学会从 τ 预测真实特征。即使预测不完美（12 维从 1 维推断本质上是欠定的），μ 也会落在合理的特征空间中。

**γ 的选择**：

| γ 值 | 效果 |
|------|------|
| 0.3 | 编码器特征预测较松散，增强样本多样性高但质量低 |
| 0.5 | 平衡点 |
| 1.0 | μ 紧贴真实特征，增强样本接近真实样本（增强效果减弱） |

**默认 γ = 0.5**，允许编码器在真实特征附近有一定发散空间。

### 5.5 KL 散度损失 L_KL（分布正则化）

$$\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{12} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

| 项目 | 说明 |
|------|------|
| 先验 | N(0, I)，12 维标准正态分布 |
| 后验 | N(μ, diag(σ²))，编码器输出的对角高斯 |
| 作用 | 防止 log σ² → -∞（方差坍缩为 0），保持采样多样性 |
| 权重 β | **0.001**，仅需最轻量的约束 |

> β = 0.001 的理由：KL 散度的主要作用在这里是防止 σ² 坍缩。因为有 L_enc 直接监督 μ，不需要 KL 来塑造 μ 的分布。β 只需足够大以避免 σ² → 0 即可。

### 5.6 重参数化技巧

```python
z = μ + ε * torch.exp(0.5 * log_var)    # ε ~ N(0, I)
```

梯度通过 μ 和 log σ² 回传，不经过随机节点 ε。

---

## 六、训练策略

### 6.1 超参数

| 超参数 | 值 | 来源 |
|--------|-----|------|
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | 复用现有 |
| Scheduler | CosineAnnealingWarmRestarts (T_0=20, T_mult=2) | 复用现有 |
| Batch Size | 4096 | 复用现有 |
| Epochs | 100 | 复用现有 |
| Early Stop Patience | 15 | 复用现有 |
| Gradient Clip | 1.0 | 复用现有 |
| Dropout | 0.15 | 复用现有 |
| α (增强权重) | **0.3** | 新增 |
| γ (编码器监督权重) | **0.5** | 新增 |
| β (KL 权重) | **0.001** | 新增 |

### 6.2 训练流程（每个 batch）

```
输入: x_true (12维), τ_true (1维, standardized)

1. 监督路径（全权重）:
   τ̂_sup = Decoder(x_true)
   L_sup = SmoothL1Loss(τ̂_sup, τ_true)          # weight = 1.0

2. 编码器前向:
   μ, log_var = Encoder(τ_true)                  # 各 12 维

3. 特征重建（编码器监督）:
   L_enc = SmoothL1Loss(μ, x_true)               # weight = γ

4. 采样 + 增强路径（降权重）:
   ε ~ N(0, I)
   z = μ + ε * exp(0.5 * log_var)               # reparameterize
   τ̂_aug = Decoder(z)
   L_aug = SmoothL1Loss(τ̂_aug, τ_true)          # weight = α

5. KL 正则化:
   L_KL = -0.5 * Σ(1 + log_var - μ² - exp(log_var))  # weight = β

6. 总损失:
   L = L_sup + α * L_aug + γ * L_enc + β * L_KL

7. backward + optimizer.step + scheduler.step
```

### 6.3 梯度流向分析

```
L_sup  ──→ Decoder parameters                    (主梯度)
L_aug  ──→ Decoder parameters                    (增强梯度)
      ──→ z ──→ μ, log_var ──→ Encoder params    (编码器间接学习)
L_enc  ──→ μ ──→ Encoder params                  (编码器直接监督)
L_KL   ──→ μ, log_var ──→ Encoder params          (方差正则化)
```

Encoder 从两个信号学习特征预测：
- **L_enc**（直接）：μ 应该接近真实特征 x_true
- **L_aug**（间接）：采样 z 应该让解码器的 τ 预测准确

这两个信号在正常情况下是一致的——接近真实的特征 → 准确的 τ 预测。当它们冲突时（例如 μ 偏离真实特征但解码器仍能准确预测 τ），L_enc 提供了"锚定"作用，防止编码器漂移到不合理区域。

### 6.4 Checkpoint

| 文件 | 内容 |
|------|------|
| `checkpoints_vae/best_model.pt` | `{"encoder": enc_state, "decoder": dec_state, ...}` |
| `checkpoints_vae/scaler.pt` | StandardScaler 的 mean 和 scale |

### 6.5 推理流程

```
x_true → Decoder → τ̂ → inverse_transform → τ̂_km
```

编码器不参与推理。推理路径与 SlabThicknessNet 完全一致。

---

## 七、与原始 SlabThicknessNet 的关系

| 维度 | SlabThicknessNet | VAE（本设计） |
|------|------------------|---------------|
| 主模型（Decoder） | SlabThicknessNet | **完全相同的 SlabThicknessNet** |
| 辅助模型 | — | Encoder（特征生成器，~62K 参数） |
| 训练数据 | 仅真实样本 | 真实样本 + 编码器采样增强 |
| 损失函数 | 仅 SmoothL1Loss | 4 项加权组合 |
| 推理路径 | features → τ̂ | **完全相同** |
| 推理开销 | 1 次前向 | **完全相同**（Encoder 不参与） |
| 可对比性 | 基线 | 可直接对比（推理时模型结构一致） |

---

## 八、我的建议

### 8.1 建议一：Encoder 预热（Warm-up）

训练初期 Encoder 的 μ 预测质量差，采样出的 z 接近噪声。此时 L_aug 不仅无益，还可能拖慢 Decoder 收敛。

**建议**：前 N 个 epoch 关闭增强路径（α=0），仅用 L_sup + L_enc + L_KL 训练。等 Encoder 的 L_enc 下降到合理水平后再开启 L_aug。

```python
α_effective = 0 if epoch < WARMUP_EPOCHS else α * min(1.0, (epoch - WARMUP_EPOCHS) / RAMP_EPOCHS)
```

默认 `WARMUP_EPOCHS=5`, `RAMP_EPOCHS=5`，即前 5 个 epoch 纯监督，之后 5 个 epoch 线性提升 α 至目标值。

### 8.2 建议二：梯度停止（可选）

如果不希望 Encoder 通过 L_aug 的反向传播被 Decoder"牵着走"（Encoder 应该学 P(features|τ)，而非"如何让 Decoder 开心"），可以在增强路径中对 z 做 detach：

```python
τ̂_aug = Decoder(z.detach())   # Encoder 不通过这条路径接收梯度
```

此时 Encoder 仅通过 L_enc + L_KL 学习，训练目标更纯粹。代价是 Encoder 与 Decoder 的协同优化被切断。

**建议默认不 detach**，如果观察到 Encoder 的 μ 输出偏离真实特征，再开启。

### 8.3 建议三：多采样增强

每个 τ_true 可以采样 K > 1 个 z，每个 z 产生一个 L_aug，取平均。这增加了增强样本的多样性。

```python
for k in range(K):
    z_k = μ + ε_k * exp(0.5 * log_var)
    τ̂_aug_k = Decoder(z_k)
    L_aug += SmoothL1Loss(τ̂_aug_k, τ_true)
L_aug /= K
```

K=2 或 3，增大有限的计算开销换取更好的覆盖。但考虑到 batch_size=4096 已经很大，K>1 可能收益递减。**建议默认 K=1**，作为可选的实验选项。

### 8.4 建议四：验证集监控

训练中同时记录以下指标，帮助判断 Encoder 是否在正常工作：

| 监控指标 | 含义 | 健康趋势 |
|----------|------|----------|
| L_sup (val) | 主任务验证损失 | 平稳下降 → 持平 |
| L_enc (val) | 编码器特征预测能力 | 稳步下降 |
| L_KL | KL 散度 | 不趋近于 0（方差未坍缩） |
| σ² 均值 | 编码器平均不确定性 | 不趋近于 0 |

如 L_KL → 0 且 σ² → 0，说明 Encoder 退化为确定性映射，增强变成"加固定偏置"，失去多样性收益。此时应增大 β。

---

## 九、代码结构

```
machine_learning_task/
├── resnet.md                       # 原残差网络设计文档（不变）
├── vae.md                          # 本文档：VAE 数据增强设计
├── run.py                          # 原入口脚本（不变）
├── run_vae.py                      # VAE 入口脚本（新增）
└── src/
    ├── config.py                   # 追加 VAE 配置项
    ├── model.py                    # SlabThicknessNet（不变，复用为 Decoder）
    ├── vae_model.py                # VAE 模型（新增：Encoder + VAE wrapper）
    ├── vae_train.py                # VAE 训练循环（新增）
    ├── data_loader.py              # 数据加载（不变，复用）
    └── evaluate.py                 # 评估（不变，复用，仅需加载不同 checkpoint）
```

### 复用清单

| 模块 | 用法 | 是否修改 |
|------|------|----------|
| `src/config.py` | 追加 VAE 配置项 | 仅末尾追加 |
| `src/model.py` / `SlabThicknessNet` | VAE 解码器 | **不修改** |
| `src/model.py` / `ResidualBlock`, `FeatureBranch` | 被 SlabThicknessNet 间接使用 | **不修改** |
| `src/data_loader.py` / `load_data()` | VAE 训练数据加载 | **不修改** |
| `src/evaluate.py` / `evaluate()` | VAE 测试评估 | **不修改**（仅加载不同 checkpoint 路径） |

---

## 十、设计决策总结

| 决策 | 选择 | 核心理由 |
|------|------|----------|
| 编码器角色 | 数据增强器（训练时辅助，推理时丢弃） | 最终目标是提升 τ 预测精度，而非构建生成模型 |
| 损失设计 | 四项加权：L_sup + α·L_aug + γ·L_enc + β·L_KL | 每项有明确分工：主任务、增强、编码器监督、分布正则 |
| 增强权重 α | 0.3 | 合成数据有信息量但不能与真实数据等权 |
| 编码器监督 γ | 0.5 | 强制 μ 学习真实特征映射，防止编码器漂移 |
| KL 权重 β | 0.001 | 仅需防止方差坍缩，不需要强分布约束 |
| 解码器 | 复用 SlabThicknessNet | 与原模型可公平对比，验证增强收益来自训练方式而非架构 |
| Encoder 预热 | 前 5 epoch 不增强 | 冷启动保护，等编码器质量达标后再开启增强 |
| 推理 | Decoder(x_true) → τ̂ | 与原模型完全一致 |
