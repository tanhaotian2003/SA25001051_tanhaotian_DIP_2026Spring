# 第三次作业
本次作业一共包括两个部分：
## Bundle Adjustment

用 PyTorch 实现 Bundle Adjustment 优化，从 2D 观测恢复：
1. **相机内参**：焦距 f（所有相机共享）
2. **每个相机的外参** (Extrinsics)：旋转 R 和平移 T（共 50 组）
3. **所有 3D 点的坐标** (X, Y, Z)（共 20000 个）

#### 具体要求：

1. **实现投影函数**：根据相机内参和外参（R, T），将 3D 点投影到 2D 像素坐标。
2. **构建优化目标**：最小化 2D 重投影误差（predicted 2D - observed 2D 的距离）。
3. **参数化与优化**：使用 Euler 角参数化旋转（推荐），使用 PyTorch 的优化器（如 Adam）进行梯度下降。
4. **可视化与评估**：展示优化过程中 loss 的变化曲线，以及最终重建的 3D 点云（保存为带颜色的 OBJ 文件，颜色从 `points3d_colors.npy` 读取）。


## COLMAP

该过程中使用COLMAP的命令行工具，对```data/image/```中的50张照片进行渲染

#### 具体步骤：

1. **特征提取** (Feature Extraction)
2. **特征匹配** (Feature Matching)
3. **稀疏重建** (Sparse Reconstruction / Mapper) — 即 COLMAP 内部的 Bundle Adjustment
4. **稠密重建** (Dense Reconstruction) — 包括 Image Undistortion、Patch Match Stereo、Stereo Fusion
5. **结果展示** — 在报告中展示稀疏点云或稠密点云的截图（可使用 [MeshLab](https://www.meshlab.net/) 查看 `.ply` 文件）

完整的命令行脚本见 [run_colmap.sh](run_colmap.sh)，可参考 [COLMAP CLI Tutorial](https://colmap.github.io/cli.html) 了解各步骤详情。

