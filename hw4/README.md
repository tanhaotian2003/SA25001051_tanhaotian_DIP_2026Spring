# Assignment 4 - Implement Simplified 3D Gaussian Splatting

### 本次实验通过纯pytorch框架补齐了简化版3DGS流程

### Resources:
- [Paper: 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf)
- [3DGS Official Implementation](https://github.com/graphdeco-inria/gaussian-splatting)
- [COLMAP — Structure-from-Motion](https://colmap.github.io/)
- [Teaching Slides](https://pan.ustc.edu.cn/share/index/66294554e01948acaf78)

---

### Background

3D Gaussian Splatting 将场景表示为一组带颜色和不透明度的 3D 高斯，通过将其投影到图像平面做 α-blending 实现可微体渲染。本作业将带你从零实现一个**简化版** 3DGS（不含 tile-based rasterizer 和 adaptive densification），完整体验 pipeline：相机参数恢复 → 3D 高斯参数化 → 投影 → α-blending。

### Data

```
data/
├── chair/images/   # 100 张 multi-view 渲染图像
└── lego/images/    # 100 张 multi-view 渲染图像
```

两个场景任选其一，下面以 `chair` 为例（你也可以用自己的多视角图像，放入 `<scene>/images/` 即可）。

---

## Task 1: Structure-from-Motion with COLMAP

使用 COLMAP 恢复相机内外参，并得到一组稀疏 3D 点作为 3DGS 的初始化：

```bash
python mvs_with_colmap.py --data_dir data/chair
```

将恢复的 3D 点重投影回各视角进行验证：

```bash
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

---

## Task 2: Simplified 3D Gaussian Splatting (主要部分)

观察 Task 1 的输出可以发现，COLMAP 恢复的 3D 点对于稠密渲染来说过于稀疏。我们将每个点扩展为一个 3D 高斯，使其覆盖周围空间。

### 2.1 3D Gaussian Initialization

参考 paper 公式 (6)：协方差矩阵由缩放矩阵 *S* 和旋转矩阵 *R* 构造。每个高斯需要以下可优化参数：

| 参数 | 说明 |
|------|------|
| Position μ | 初始化为 SfM 3D 点 |
| Rotation R | 用单位四元数参数化 |
| Scaling S | 3 维向量 |
| Opacity o | 标量 |
| Color c | RGB 三通道 |

[gaussian_model.py#L32](gaussian_model.py#L32) 已实现这些参数的初始化。

> **本次实验已完成**:在 [gaussian_model.py#L103](gaussian_model.py#L103) 中由四元数和缩放参数构造 **3D 协方差矩阵**。

### 2.2 Project 3D Gaussians to 2D

参考 paper 公式 (5)，将 3D 高斯投影到图像平面需要：

- 世界到相机的变换矩阵 *W*
- 投影变换的雅可比矩阵 *J*

投影后的 2D 协方差为 $\Sigma' = J W \Sigma W^T J^T$。

> **本次实验已完成**：在 [gaussian_renderer.py#L26](gaussian_renderer.py#L26) 中实现了 3D → 2D 投影。

### 2.3 Compute 2D Gaussian Values

2D Gaussian 在像素 $\mathbf{x}$ 处的取值：

$$
f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{2\pi\sqrt{|\boldsymbol{\Sigma}_i|}} \exp\left(P_{(\mathbf{x},i)}\right), \quad P_{(\mathbf{x},i)} = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)
$$

其中 **μᵢ** 与 **Σᵢ** 为投影后的 2D 高斯中心与协方差。

> **TODO**：在 [gaussian_renderer.py#L61](gaussian_renderer.py#L61) 中计算 Gaussian 取值。

### 2.4 Volume Rendering via α-blending

给定 *N* 个按深度排序的 2D 高斯，每个高斯在像素 $\mathbf{x}$ 处的 alpha 与透射率为：

$$
\alpha_{(\mathbf{x}, i)} = o_i \cdot f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i), \qquad T_{(\mathbf{x}, i)} = \prod_{j<i} (1 - \alpha_{(\mathbf{x}, j)})
$$

最终像素颜色由各高斯按 α-blending 累加（paper 公式 1-3）。

> **本次实验已完成**：在 [gaussian_renderer.py#L83](gaussian_renderer.py#L83) 中实现最终渲染。

### Train your 3DGS

完成上述代码后，启动训练：

```bash
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

### Render a Multi-view Video (Optional)

训练完成后，可用 [render_3dgs_mv.py](render_3dgs_mv.py) 沿一个绕场景中心的**水平圆轨迹**渲染一段连续视角视频，便于直观检查重建质量：

```bash
python render_3dgs_mv.py \
    --colmap_dir data/chair \
    --checkpoint data/chair/checkpoints/checkpoint_000060.pt \
    --num_frames 240 --fps 30
# 默认输出: <colmap_dir>/render_mv.mp4
```

up 轴由训练相机的 y 轴平均自动估计（NeRF 合成数据图像均为正放），orbit 半径与高度取训练相机的均值。

---

## Task 3: Compare with the Official 3DGS Implementation

本作业为纯 PyTorch 实现，训练速度与显存效率远不如官方实现，且未实现 adaptive Gaussian densification 等关键模块。请使用相同数据集运行 [官方 3DGS](https://github.com/graphdeco-inria/gaussian-splatting)，从**渲染质量、训练速度、显存占用**三方面进行对比。

# 🎨 1. 渲染质量对比（Rendering Quality）

## ✔ 官方 3DGS

官方实现能够生成：

- 细节清晰的高质量重建结果
- 几何边界锐利（edge sharpness 高）
- 纹理区域具有良好的高频细节表达
- 多视角一致性较强（view consistency）

## ✔ 本次 PyTorch 实现

本次简化实现的结果表现为：

- 图像整体略微模糊
- 高频细节（如边缘、纹理）存在一定损失
- 在部分视角下可能出现轻微“漂浮点”（floaters）
- 重建质量依赖初始化点质量较大

---

## 🧠 原理分析（关键差异来源）

### ⭐ 1. 是否具备 Adaptive Densification（自适应密度控制）

- 官方方法：
  - 根据重建误差动态**增加/分裂 Gaussian**
  - 在高误差区域（边缘、纹理）提高表示能力
  - 实现“非均匀建模能力分配”

- 本次实现：
  - Gaussian 数量固定
  - 无法根据场景复杂度调整密度
  - 表达能力受限

👉 结论：  
官方方法在复杂区域具有更高表达能力，因此细节更清晰。

---

### ⭐ 2. 表示能力差异

Gaussian Splatting 的本质是：

> 用有限数量的高斯去拟合连续场景函数

当 Gaussian 数量固定时：

- 平滑区域 → 表现良好
- 高频区域 → 表达不足（信息欠拟合）

---

# ⚡ 2. 训练速度对比（Training Speed）

## ✔ 官方 3DGS

- 可达到接近实时训练/渲染
- GPU utilization 高
- iteration throughput 高

## ✔ 本次 PyTorch 实现

- 训练速度明显较慢
- 单 iteration 耗时较长
- GPU 利用率较低

---

## 🧠 原理分析（核心瓶颈）

### ⭐ 1. CUDA vs PyTorch 实现差异

官方方法使用：

- 自定义 CUDA rasterization kernel
- tile-based 并行 splatting
- kernel fusion（减少 memory IO）

👉 复杂度优化：

\[
O(N \times H \times W) \rightarrow \text{高度并行优化}
\]

---

本次实现：

- 使用 PyTorch tensor 运算
- 存在 Python 级循环（alpha blending）
- 未进行 kernel fusion

👉 实际复杂度：

\[
O(N \times H \times W)
\]

---

### ⭐ 2. Rasterization 方式差异

- 官方：
  - GPU tile-based rendering（局部区域并行计算）

- 本实现：
  - 全图逐 Gaussian 计算

👉 导致计算开销显著增加

---

# 💾 3. 显存占用对比（Memory Usage）

## ✔ 官方 3DGS

- 显存占用较低
- 支持 large-scale scene
- Gaussian 可动态裁剪

## ✔ 本次 PyTorch 实现

- 显存占用较高
- 中间张量较多
- 无 pruning / densification

---

## 🧠 原理分析

### ⭐ 1. 是否使用 Sparse Gaussian 管理

- 官方：
  - 会 prune 不重要 Gaussian
  - 只保留有效 contribution region

- 本实现：
  - 所有 Gaussian 始终参与计算
  - 产生完整 (N × H × W) 中间 tensor

---

### ⭐ 2. 中间变量存储方式

本实现中存在：

- Gaussian value map (N, H, W)
- alpha map (N, H, W)
- weight map (N, H, W)

👉 导致显存复杂度：

\[
O(N \times H \times W)
\]

---

# 📌 4. 总体差异总结

三者差异可以归因于以下三个核心因素：

## ⭐ (1) 缺少 Adaptive Densification
- 固定 Gaussian 数量
- 表达能力不足

## ⭐ (2) 缺少 CUDA 优化 Rasterizer
- 无 tile-based rendering
- 无 kernel fusion
- Python/PyTorch overhead 大

## ⭐ (3) 缺少 Sparse Memory Management
- 无 pruning
- 无动态 Gaussian 管理
- 中间张量占用显存较高

---

# 🎯 5. 核心结论（可用于总结段）

官方 3D Gaussian Splatting 的性能优势并不仅来源于其高斯表示本身，而更关键在于其完整的工程优化设计，包括：

- 自适应 Gaussian densification（提升表达能力）
- CUDA 加速的 tile-based rasterization（提升训练速度）
- 稀疏化与动态剪枝机制（降低显存占用）

相比之下，本次 PyTorch 实现虽然完整复现了 3DGS 的数学建模流程，但由于缺乏上述工程级优化，因此在性能上存在明显差距。



---

### Requirements:
- 请自行环境配置，推荐使用 [conda 环境](https://docs.anaconda.com/miniconda/)


