import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pytorch3d.transforms import euler_angles_to_matrix

# --- 模块 1: 数据加载 (Data Loading) ---
def load_data(data_path="data"):
    # 加载 2D 观测点和 visibility (50, 20000, 3)
    points2d_data = np.load(os.path.join(data_path, "points2d.npz"))
    # 加载每个点的预设颜色 (20000, 3)
    colors = np.load(os.path.join(data_path, "points3d_colors.npy"))
    
    all_obs, all_vis = [], []
    for i in range(50):
        view_data = points2d_data[f"view_{i:03d}"]
        all_obs.append(view_data[:, :2])
        all_vis.append(view_data[:, 2])
        
    return (torch.tensor(np.array(all_obs), dtype=torch.float32), 
            torch.tensor(np.array(all_vis), dtype=torch.float32), 
            torch.tensor(colors, dtype=torch.float32))

# --- 模块 2: BA 模型定义 (Parameters & Projection) ---
class BAModel(nn.Module):
    def __init__(self, num_views=50, num_points=20000):
        super().__init__()
        # 优化目标 A: 相机内参 (焦距)
        self.focal_length = nn.Parameter(torch.tensor(1000.0))
        
        # 优化目标 B: 相机外参 (50组欧拉角和平移)
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3))
        self.translations = nn.Parameter(torch.tensor([[0.0, 0.0, -3.0]]).repeat(num_views, 1))
        
        # 优化目标 C: 3D 点云坐标 (20000个点)
        self.points3d = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        
    def forward(self, cx=512, cy=512):
        # 1. 旋转矩阵转换
        R = euler_angles_to_matrix(self.euler_angles, convention="XYZ")
        
        # 2. 世界坐标转相机坐标: P_cam = R @ P_world + T
        pts_world = self.points3d.unsqueeze(0).unsqueeze(-1)
        pts_cam = torch.matmul(R.unsqueeze(1), pts_world) + self.translations.unsqueeze(1).unsqueeze(-1)
        pts_cam = pts_cam.squeeze(-1) # (50, 20000, 3)
        
        # 3. 透视投影 (核心公式)
        Xc, Yc, Zc = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
        eps = 1e-8
        u = -self.focal_length * Xc / (Zc + eps) + cx
        v = self.focal_length * Yc / (Zc + eps) + cy
        
        return torch.stack([u, v], dim=-1)

# --- 模块 3: 训练逻辑 (Optimization Loop) ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_2d, vis_mask, colors = load_data()
    obs_2d, vis_mask = obs_2d.to(device), vis_mask.to(device)
    
    model = BAModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    for step in range(2001):
        pred_2d = model()
        # 只计算可见点的重投影误差 (Reprojection Error)
        loss = (torch.sum((pred_2d - obs_2d)**2, dim=-1) * vis_mask).sum() / vis_mask.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    save_obj(model.points3d.detach().cpu().numpy(), colors.numpy(), "reconstruction.obj")

def save_obj(points, colors, filename):
    with open(filename, 'w') as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

if __name__ == "__main__":
    train()