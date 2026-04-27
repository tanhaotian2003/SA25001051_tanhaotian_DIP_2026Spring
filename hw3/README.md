# 第三次作业
本次作业一共包括两个部分：
## Bundle Adjustment

## COLMAP

该过程中使用COLMAP的命令行工具，对```data/image/```中的50张照片进行渲染

#### 具体步骤：

1. **特征提取** (Feature Extraction)
2. **特征匹配** (Feature Matching)
3. **稀疏重建** (Sparse Reconstruction / Mapper) — 即 COLMAP 内部的 Bundle Adjustment
4. **稠密重建** (Dense Reconstruction) — 包括 Image Undistortion、Patch Match Stereo、Stereo Fusion
5. **结果展示** — 在报告中展示稀疏点云或稠密点云的截图（可使用 [MeshLab](https://www.meshlab.net/) 查看 `.ply` 文件）

完整的命令行脚本见 [run_colmap.sh](run_colmap.sh)，可参考 [COLMAP CLI Tutorial](https://colmap.github.io/cli.html) 了解各步骤详情。

