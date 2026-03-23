#run_point_transform_v2.py
import cv2
import numpy as np
import gradio as gr

# 全局变量，用于存储源控制点和目标控制点
points_src = []
points_dst = []
image = None

# 当上传新图像时重置控制点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    # 转换为 numpy 数组确保后续 OpenCV 处理正常
    image = np.array(img)
    return img

# 记录点击的点并在图像上可视化
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return None
    
    x, y = evt.index[0], evt.index[1]

    # 交替记录起点和终点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # 绘制点（蓝色：起点，红色：终点）和箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)  # 蓝色起点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)  # 红色终点

    # 绘制从起点到终点的绿色箭头
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image

# 基于点的图像变形核心函数
def point_guided_deformation(image, source_pts, target_pts, alpha=2.0, eps=1e-8):
    """
    使用反距离加权 (Inverse Distance Weighting) 实现局部变形
    """
    if image is None or len(source_pts) == 0:
        return image

    h, w = image.shape[:2]
    
    # 1. 生成目标图像的网格坐标
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    target_grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)

    if len(source_pts) < 1:
        return image

    # 2. 计算权重
    # 计算网格中每个点到所有目标控制点的距离
    # 使用 numpy 广播机制加速计算
    diff = target_grid[:, np.newaxis, :] - target_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # 计算权重 w = 1 / (d^alpha + eps)
    weights = 1.0 / (np.power(dist, alpha) + eps)
    # 归一化权重，使每个像素受到的总权重为 1
    weights /= np.sum(weights, axis=1, keepdims=True)

    # 3. 计算位移
    displacements = source_pts - target_pts
    grid_displacements = np.sum(weights[:, :, np.newaxis] * displacements[np.newaxis, :, :], axis=1)
    
    # 4. 反向映射：目标坐标 + 位移 = 对应的原图坐标
    source_grid = target_grid + grid_displacements
    
    # 重塑为 remap 所需的映射矩阵
    map_x = source_grid[:, 0].reshape(h, w).astype(np.float32)
    map_y = source_grid[:, 1].reshape(h, w).astype(np.float32)

    # 5. 重采样生成变形后的图像
    warped_image = cv2.remap(
        image, 
        map_x, 
        map_y, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REFLECT
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image
    if image is None or not points_src or not points_dst:
        return image

    # 执行变形
    warped_image = point_guided_deformation(
        image, 
        np.array(points_src), 
        np.array(points_dst)
    )

    return warped_image

# 清除所有选择的点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## 基于点对引导的图像局部变形 (Point-Guided Warping)")
    gr.Markdown("使用说明：在左二图中点击。**第1次点击为起点，第2次为终点**，以此类推。完成后点击 Run。")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="1. 上传图片", interactive=True)
            point_select = gr.Image(label="2. 点击设置变形点对", interactive=True)

        with gr.Column():
            result_image = gr.Image(label="3. 变形结果")

    with gr.Row():
        run_button = gr.Button("🚀 Run Warping", variant="primary")
        clear_button = gr.Button("🧹 Clear Points")

    # 事件绑定
    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

if __name__ == "__main__":
    demo.launch()