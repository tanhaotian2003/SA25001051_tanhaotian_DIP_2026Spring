import torch
from torch.utils.data import Dataset
import cv2

class FacadesDataset(Dataset):
    def __init__(self, list_file):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
       # img_name = self.image_filenames[idx]
       # img_color_semantic = cv2.imread(img_name)
        # Convert the image to a PyTorch tensor
       # image = torch.from_numpy(img_color_semantic).permute(2, 0, 1).float()/255.0 * 2.0 -1.0
       # image_rgb = image[:, :, :256]
       # image_semantic = image[:, :, 256:]
        #return image_rgb, image_semantic
        # 1. 读取原图
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        
        # 2. 这里的图片是左右拼接的，先拿到总宽度
        h, w, _ = img_color_semantic.shape
        half_w = w // 2
        
        # 3. 切割 A 和 B
        img_A = img_color_semantic[:, :half_w, :]
        img_B = img_color_semantic[:, half_w:, :]
        
        # 4. 强制 Resize 到 256x256 (关键步骤！)
        img_A = cv2.resize(img_A, (256, 256))
        img_B = cv2.resize(img_B, (256, 256))
        
        # 5. 归一化并转 Tensor
        # 记得我们之前的归一化逻辑 [-1, 1]
        img_A = torch.from_numpy(img_A).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        img_B = torch.from_numpy(img_B).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        
        return img_A, img_B