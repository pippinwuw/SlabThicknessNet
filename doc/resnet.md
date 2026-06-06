# 电离层等效板厚预测 — 残差多分支神经网络设计

## 一、任务背景与目标

电离层等效板厚（Equivalent Slab Thickness, τ）是描述电离层等离子体垂直分布的关键参数，定义为：

$$\tau = \frac{\text{vTEC}}{\text{NmF2}}$$

即垂直总电子含量与 F2 层峰值电子密度的比值。准确预测 τ 对卫星导航、短波通信和天波超视距雷达至关重要。

**任务**：使用 PyTorch 构建神经网络替代论文中的 XGBoost+EL 方法，在给定 12 维空间-时间-太阳物理特征下预测 τ (km)。

**论文基线**（XGBoost + Ensemble Learning）：

| 指标 | 值 |
|------|-----|
| RMSE | 62.3 km |
| MAE  | 41.5 km |
| MAPE | 13.1% |
| R    | 0.904 |

---

## 二、数据与特征工程

### 2.1 数据概览

| 项目 | 说明 |
|------|------|
| 总条数 | ~217 万条 |
| 时间跨度 | 2006–2020 年 |
| 数据源 | GNSS 掩星观测 |
| 划分 | subset 列含 train / val / test |
| 目标 | tau_km，范围 ~100–700 km，均值 ~354 km |

### 2.2 输入特征（12 维）

特征已由原始论文完成预处理（归一化、周期编码），无需额外工程。

| 分组 | 特征 | 物理含义 | 值域 | 编码方式 |
|------|------|----------|------|----------|
| **空间 (4)** | `proc_sin_lon` | 经度正弦 | [-1, 1] | sin(lon) |
| | `proc_cos_lon` | 经度余弦 | [-1, 1] | cos(lon) |
| | `proc_lat` | 地理纬度 | [0, 1] | MinMax 归一化 |
| | `proc_mlat` | 修正磁纬度 | [0, 1] | MinMax 归一化 |
| **太阳/电离层 (3)** | `proc_kp` | Kp 地磁活动指数 | [0, 1] | MinMax 归一化 |
| | `proc_f107` | F10.7 cm 太阳射电通量 | [0, 1] | MinMax 归一化 |
| | `proc_vtec` | 垂直总电子含量 | [0, 1] | MinMax 归一化 |
| **时间 (5)** | `proc_sin_lt` | 地方时正弦 | [-1, 1] | sin(LT) |
| | `proc_cos_lt` | 地方时余弦 | [-1, 1] | cos(LT) |
| | `proc_sin_doy` | 年积日正弦 | [-1, 1] | sin(DOY) |
| | `proc_cos_doy` | 年积日余弦 | [-1, 1] | cos(DOY) |
| | `proc_cos_chi` | 太阳天顶角余弦 | [0, 1] | cos(SZA) |

> **特征顺序约束**：spatial(4) → solar(3) → temporal(5) = 12 维。此顺序与 `model.py` 中列切片逻辑严格对应，不可随意调整。

### 2.3 输出与预处理

| 项目 | 说明 |
|------|------|
| 原始目标 | `tau_km`，物理单位 km，范围 ~100–700 |
| 训练目标 | 经 `StandardScaler` 变换为 N(0, 1) |
| 输出层 | 1 神经元，无激活函数（线性） |
| 推理时 | 逆变换回 km 单位 |

**为什么用 StandardScaler 而非 MinMax？** τ 的分布接近正态但带右尾（高 τ 值对应低太阳活动期）。StandardScaler 保持相对距离关系，不会像 MinMax 那样被极端值压缩中间区域的区分度。

---

## 三、网络架构设计

### 3.1 设计动机

电离层板厚受三类物理过程驱动，它们的特征形态和时间尺度各异：

| 驱动因素 | 特征 | 变化尺度 |
|----------|------|----------|
| 空间位置 | 经纬度、磁纬度 | 静态/缓变（随卫星轨道移动） |
| 太阳-电离层耦合 | vTEC、Kp、F10.7 | 小时～天级（受太阳活动和地磁扰动控制） |
| 时间周期 | 地方时、季节、太阳天顶角 | 严格周期性（日变化、年变化） |

直觉上，让网络先在各模态内部分别提取表征，再融合，比将所有特征一股脑输入全连接层更能利用这种物理结构。这就是 **多分支（Multi-Branch）** 设计的出发点。

同时，板厚与特征之间存在复杂的非线性耦合（如 vTEC 的影响随太阳活动水平变化），需要足够深度来建模。但深层全连接网络易退化——残差连接是应对此问题的成熟方案。

### 3.2 整体架构

```
                       Input (12)
                          │
             ┌────────────┴────────────┐
             │                         │
    ┌────────▼────────┐      ┌────────▼────────┐
    │ 物理分支         │      │ 时间分支         │
    │ spatial(4)+      │      │ temporal(5)      │
    │ solar(3) = 7     │      │ = 5              │
    │                  │      │                  │
    │ Linear(7→128)    │      │ Linear(5→64)     │
    │   +LayerNorm     │      │   +LayerNorm     │
    │   +GELU+Dropout  │      │   +GELU+Dropout  │
    │ Linear(128→256)  │      │ Linear(64→128)   │
    │   +LayerNorm     │      │   +LayerNorm     │
    │   +GELU+Dropout  │      │   +GELU+Dropout  │
    └────────┬─────────┘      └────────┬─────────┘
             │            concat          │
             └──────────┬─────────────────┘
                        │ (256 + 128 = 384)
                        ▼
              Fusion: Linear(384→384) + LayerNorm + GELU
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼           ▼
      ResBlock 1   ResBlock 2   ResBlock 3   ResBlock 4
      384→512      512→512      512→256      256→128
      skip:proj    skip:id      skip:proj    skip:proj
            │           │           │           │
            └───────────┴───────────┴───────────┘
                        │
                        ▼
              Head: 128 → 64 → 32 → 1  (输出 τ)
```

### 3.3 分支划分理由

| 分支 | 输入 | 设计理由 |
|------|------|----------|
| **物理分支** | spatial(4) + solar(3) = 7 维 | 空间位置决定电离层背景状态，太阳活动（Kp、F10.7）和 vTEC 直接控制电离率。这些变量在物理上是耦合的：vTEC 的绝对水平取决于太阳辐射（F10.7）和地磁扰动（Kp），而它们的效应随地理位置变化。放在同一分支让网络可以学习它们的交互。 |
| **时间分支** | temporal(5) = 5 维 | 地方时和季节的周期性变化（日变化、年变化）是叠加在背景状态上的调制信号。用一个独立、较窄（64→128）的分支迫使网络以紧凑的方式学习时间模式，避免与空间特征过早混合。 |

> 备选：如果实验表明分支拆分无显著增益，可回退到单塔 MLP，但在物理上分支设计更有解释性。

### 3.4 残差块（ResidualBlock）

```
     input (in_dim)
        │
        ├── proj(skip) ──────────────────────────┐
        │   (1×1 Linear, 仅当 in_dim≠out_dim)      │
        │                                          ├─ + ─→ GELU ─→ output
        └── Linear(in→hidden, bias=False) ──→     │
              LayerNorm(hidden)                    │
              GELU                                 │
              Dropout(p)                           │
              Linear(hidden→out, bias=False) ──→   │
              LayerNorm(out) ──────────────────────┘
```

**关键设计选择：**

| 选择 | 理由 |
|------|------|
| **Pre-activation 风格** | 借鉴 ResNet-v2：在加法后应用 GELU，使跳过连接的梯度路径完全无阻碍 |
| **LayerNorm 而非 BatchNorm** | BatchNorm 在小批次或特征分布差异大（我们的特征有的 ∈[0,1]，有的 ∈[-1,1]）时不稳定。LayerNorm 沿特征维度归一化，不受 batch size 影响，更适合回归任务 |
| **bias=False + LayerNorm** | LayerNorm 自带 affine 参数（γ, β），Linear 的 bias 是冗余的。去掉减少 ~1% 参数量 |
| **Kaiming Normal 初始化** | 配合 GELU（类 ReLU 激活函数），保证各层输出方差稳定，避免训练初期梯度消失/爆炸 |

### 3.5 残差连接的必要性

随着网络加深（此处 4 个残差块 + 分支 + Head ≈ 12 层 Linear），普通 MLP 面临：

1. **梯度消失**：反向传播时梯度逐层衰减，浅层几乎不更新
2. **退化问题**：深层网络的训练误差反而不如浅层（不是过拟合）

残差连接提供了"恒等映射高速公路"，网络每层只需学习残差 F(x) = H(x) − x。当恒等映射已足够时，网络只需将 F(x) 推向零——这对梯度下降来说比直接拟合 H(x) = x 容易得多。

对于板厚预测，虽然 vTEC 是主要预测因子（τ = vTEC / NmF2 定义了物理关系），但 NmF2 未直接作为特征输入——网络需要从其他代理特征（太阳活动、时间、位置）中间接推断，需要足够的网络容量和深度。

### 3.6 参数统计

| 模块 | 构成 | 参数量 |
|------|------|--------|
| 物理分支 | Linear(7→128) + Linear(128→256) | ~38K |
| 时间分支 | Linear(5→64) + Linear(64→128) | ~14K |
| Fusion | Linear(384→384) | ~148K |
| ResBlock 1 | 384→512→512 + proj(384→512) | ~461K |
| ResBlock 2 | 512→512→512 (identity skip) | ~525K |
| ResBlock 3 | 512→256→256 + proj(512→256) | ~328K |
| ResBlock 4 | 256→128→128 + proj(256→128) | ~82K |
| Head | 128→64→32→1 | ~12K |
| **总计** | | **~1.88M** |

约 188 万参数，对于 217 万训练样本来说，不会过拟合（参数量 < 样本量），同时有足够容量学习非线性映射。

---

## 四、损失函数设计

### 4.1 问题分析

回归任务常用的 MSE Loss 对异常值极其敏感。分析 τ 的分布：

| 属性 | 说明 |
|------|------|
| 值域 | ~100–700 km |
| 均值 | ~354 km |
| 尾部 | 右偏（极端太阳活动/高纬地区的观测噪声较大） |
| 论文中 y_model | 用 log 变换处理（`raw_target = ln(tau)`），说明原始 τ 存在偏态 |

如果直接用 MSE，误差平方项 `(ŷ − y)²` 会使少数大误差样本主导梯度更新，导致：
- 模型偏向高 τ 区域（因为误差绝对值更大）
- 训练不稳定，验证损失震荡

### 4.2 SmoothL1Loss (Huber Loss)

选用 `torch.nn.SmoothL1Loss(β=1.0)`，其定义为：

$$\text{SmoothL1}(x, y) = \begin{cases} 0.5 \cdot (x - y)^2 / \beta & \text{if } |x - y| < \beta \\ |x - y| - 0.5 \cdot \beta & \text{otherwise} \end{cases}$$

即：当预测误差 < β（在我们的标准化空间中 ≈ 1σ）时用 MSE（平滑、梯度连续），当误差 > β 时切换为 MAE（对异常值不敏感）。

| 对比维度 | MSE | MAE | SmoothL1Loss ✓ |
|----------|-----|-----|-----------------|
| 小误差梯度 | ∝ error（无界） | constant（稀疏） | ∝ error（平滑） |
| 大误差梯度 | ∝ error（过大） | constant（有界） | constant（有界） |
| x=0 处可导 | ✓ | ✗（不可导） | ✓ |
| 对异常值鲁棒 | ✗ | ✓ | ✓ |

**选择理由**：
1. **物理约束**：vTEC 测量在高纬度和磁暴期间有较大观测误差，导致部分 τ 标签本身不可靠。MAE 在大误差区间的恒定梯度防止这些样本主导训练
2. **梯度平滑**：在零点附近可导避免了 MAE 的不稳定性——AdamW 依赖梯度的一/二阶矩估计，不可导点会引入噪声
3. **β=1.0**：配合 StandardScaler 后的目标 N(0,1)，约 68% 的样本落在 |error| < 1 范围内使用 MSE 分支，约 32% 的尾部和异常值使用 MAE 分支

### 4.3 备选方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **MSE** | 收敛快，梯度平滑 | 对异常值极度敏感 | 标签无噪声的纯物理模拟数据 |
| **MAE** | 对异常值鲁棒 | 零点不可导，收敛慢 | 噪声极大的观测数据 |
| **SmoothL1Loss ✓** | 兼具两者优点 | 多一个 β 超参 | **我们的场景** |
| **Log-Cosh** | 全定义域可导 | 对极大误差仍有二次增长 | MSE 的平滑替代 |
| **Huber(δ=0.5)** | 更早切换 MAE | 大部分样本用 MAE，收敛慢 | 异常值比例高时 |

### 4.4 可扩展的损失函数设计

若实验结果不理想，可进一步考虑：

**加权组合损失**：
$$\mathcal{L} = \lambda_1 \cdot \text{MSE} + \lambda_2 \cdot \text{PhysicsInformedLoss}$$

其中物理约束项利用 τ 的定义式 τ = vTEC / NmF2（NmF2 在数据中以 `raw_nmf2` 给出），惩罚与物理公式不一致的预测。

**分位数损失（Quantile Loss）**：若需要不确定性估计，可同时输出 τ 的预测区间（如 10%/50%/90% 分位数）。

---

## 五、训练策略

### 5.1 超参数配置

| 超参数 | 值 | 理由 |
|--------|-----|------|
| Optimizer | **AdamW** | Adam 的自适应学习率 + 解耦的权重衰减。比 SGD 收敛更快，比 Adam 泛化更好 |
| LR | 1e-3 | Adam 的推荐初始值 |
| Weight Decay | 1e-4 | 轻量 L2 正则化，防止过拟合 |
| Scheduler | **CosineAnnealingWarmRestarts** | T_0=20 epoch 后热重启。周期性提高 LR 可以帮助跳出局部极小值，且不需要额外调参 |
| Batch Size | **4096** | GPU 显存允许的最大值。大 batch 稳定梯度估计 |
| Epochs | 100 | 配合 early stopping (patience=15)，通常在 30–50 epoch 处收敛 |
| Gradient Clip | 1.0 | 梯度范数裁剪，防止某个 batch 的异常样本造成参数跳变 |
| Dropout | 0.15 | 轻量 dropout，防止神经元共适应 |

### 5.2 数据处理流程

```
CSV 全量 (217万)
    │
    ├─ subset == 'train' ──→ 采样 200K 行 (快速验证) / 全量 (最终训练)
    │                         │
    │                         ├─ target: StandardScaler.fit_transform(y)
    │                         ├─ 随机划分 9:1 → train / val
    │                         └─ DataLoader(batch_size=4096, shuffle=True)
    │
    ├─ subset == 'val' ────→ (暂不使用，训练内自划 val)
    │
    └─ subset == 'test' ───→ 最终评估
                              target: StandardScaler.transform(y)  # 使用 train 上 fit 的参数
                              逆变换预测值回 km 单位
```

### 5.3 训练监控

- **Early Stopping**：验证 loss 连续 15 epoch 不下降则停止，恢复最佳权重
- **Checkpoint**：每次新最优 val_loss 时保存 `best_model.pt`
- **Scaler 持久化**：保存 `scaler.pt`（mean, scale），推理时使用

---

## 六、评估体系

### 6.1 指标定义

| 指标 | 公式 | 单位 | 含义 |
|------|------|------|------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(\hat{y}_i - y_i)^2}$ | km | 均方根误差。对大误差惩罚重，反映最坏情况 |
| **MAE** | $\frac{1}{n}\sum|\hat{y}_i - y_i|$ | km | 平均绝对误差。反映典型误差水平 |
| **MAPE** | $\frac{100}{n}\sum|\frac{\hat{y}_i - y_i}{y_i}|$ | % | 相对误差。在 τ 跨度大时（100–700km）比绝对误差更有意义 |
| **R²** | $1 - \frac{\sum(\hat{y}_i - y_i)^2}{\sum(y_i - \bar{y})^2}$ | — | 决定系数。1 为完美预测，0 为仅预测均值 |
| **R** | $\frac{\text{Cov}(\hat{y}, y)}{\sigma_{\hat{y}}\sigma_y}$ | — | 皮尔逊相关系数。衡量线性相关性 |

### 6.2 目标

**RMSE < 62.3 km**，超越论文 XGBoost+EL 基线。

---

## 七、代码结构

```
machine_learning_task/
├── dataset.txt                    # 数据集字段说明
├── paper_summary.txt              # 参考论文摘要
├── model_input_with_raw.csv       # 原始数据集 (~718 MB)
├── resnet.md                      # 本文档：网络设计说明
├── run.py                         # 入口脚本
└── src/
    ├── __init__.py                # 包初始化
    ├── config.py                  # 全局超参数 & 特征分组
    ├── data_loader.py             # CSV 读取 → Dataset → DataLoader
    ├── model.py                   # SlabThicknessNet + ResidualBlock + FeatureBranch
    ├── train.py                   # 训练循环 + early stopping + checkpoint
    └── evaluate.py                # 评估指标 + 与基线对比
```

---

## 八、设计决策总结

| 决策 | 选择 | 核心理由 |
|------|------|----------|
| 基础结构 | Residual MLP | 需要深度建模非线性但避免退化 |
| 分支策略 | 双分支（物理+时间） | 利用物理先验，特征分组提取 |
| 归一化 | LayerNorm | 不受 batch size 影响，适合特征分布差异大的场景 |
| 激活函数 | GELU | 比 ReLU 更平滑，适合回归 |
| 损失函数 | SmoothL1Loss (β=1.0) | 平衡 MSE 的梯度平滑和 MAE 的异常值鲁棒 |
| 优化器 | AdamW | 自适应学习率 + 解耦 weight decay |
| 调度器 | CosineAnnealingWarmRestarts | 热重启逃逸局部极小值 |
| Target 预处理 | StandardScaler | 保持分布相对关系，配合 SmoothL1Loss 的 β=1.0 |