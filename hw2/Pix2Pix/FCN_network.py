# FCN_network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        # 3. 64x64 -> 32x32
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        # 4. 32x32 -> 16x16 (瓶颈层)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        # --- Decoder (上采样过程) ---
        # 1. 16x16 -> 32x32
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        # 2. 32x32 -> 64x64
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        # 3. 64x64 -> 128x128
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # 4. 128x128 -> 256x256 (最后一层输出 3 通道 RGB)
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            # 使用 Tanh 将像素值映射到 [-1, 1]，这符合 Pix2Pix 论文的建议
            nn.Tanh() 
        )
    def forward(self, x):
        ### FILL: encoder-decoder forward pass
        # Encoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        # Decoder forward pass
        d1 = self.deconv1(x4)
        d1 = F.interpolate(d1, size=(x3.size(2), x3.size(3)), mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, x3], dim=1)
        
        d2 = self.deconv2(d1)
        # 强制将 d2 的大小调整为 x2 的大小
        d2 = F.interpolate(d2, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, x2], dim=1)
        
        d3 = self.deconv3(d2)
        # 强制将 d3 的大小调整为 x1 的大小
        d3 = F.interpolate(d3, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, x1], dim=1)
        
        output = self.deconv4(d3)
        # 最终输出强制对齐回输入 x 的大小 (256x256)
        output = F.interpolate(output, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return output
    