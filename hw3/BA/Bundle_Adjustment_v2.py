import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt  # 新增：用于绘图
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
        self.focal_length = nn.Parameter(torch.tensor(1000.0))
        self.euler_angles = nn.Parameter(torch.zeros(num_views, 3))
        self.translations = nn.Parameter(torch.tensor([[0.0, 0.0, -3.0]]).repeat(num_views, 1))
        self.points3d = nn.Parameter(torch.randn(num_points, 3) * 0.1)
        
    def forward(self, cx=512, cy=512):
        R = euler_angles_to_matrix(self.euler_angles, convention="XYZ")
        pts_world = self.points3d.unsqueeze(0).unsqueeze(-1)
        pts_cam = torch.matmul(R.unsqueeze(1), pts_world) + self.translations.unsqueeze(1).unsqueeze(-1)
        pts_cam = pts_cam.squeeze(-1) 
        
        Xc, Yc, Zc = pts_cam[..., 0], pts_cam[..., 1], pts_cam[..., 2]
        eps = 1e-8
        u = -self.focal_length * Xc / (Zc + eps) + cx
        v = self.focal_length * Yc / (Zc + eps) + cy
        
        return torch.stack([u, v], dim=-1)

# --- 模块 3: 训练与可视化逻辑 ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_2d, vis_mask, colors = load_data()
    obs_2d, vis_mask = obs_2d.to(device), vis_mask.to(device)
    
    model = BAModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    # --- 新增：用于记录 Loss 的列表 ---
    loss_history = []
    
    print("🚀 Starting Bundle Adjustment Optimization...")
    for step in range(2001):
        pred_2d = model()
        loss = (torch.sum((pred_2d - obs_2d)**2, dim=-1) * vis_mask).sum() / vis_mask.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录 Loss
        loss_history.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

    # --- 新增：绘制并保存 Loss 曲线 ---
    plot_loss(loss_history)
    
    save_obj(model.points3d.detach().cpu().numpy(), colors.numpy(), "reconstruction.obj")
    print("✅ Reconstruction finished. Output: reconstruction.obj & loss_curve.png")

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Reprojection Error', color='#2ca02c', linewidth=2)
    plt.yscale('log')  # 使用对数坐标，效果更好
    plt.title('Bundle Adjustment Optimization Convergence', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Loss (MSE - Log Scale)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300) # 高清保存
    plt.close()

def save_obj(points, colors, filename):
    with open(filename, 'w') as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

if __name__ == "__main__":
    train()