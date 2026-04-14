# 实验描述

本实验要求仿照`CVPR17`的论文`Image-to-Image Translation with Conditional Adversarial Nets` 实现基于全卷积网络(FCN)的图像到图像的翻译(需要注意原文是使用的GAN模型)。Github源代码是在`facades`数据集上训练的，本实验要求在`Maps`数据集上训练。
目标在 `Maps` 数据集上完成从“卫星图”到“地图”的自动生成任务。

## 实验设计

``硬件:``本实验在USTC3DV的 Ubuntu服务器提供的NVIDIA RTX 4090 (Node11)上进行。

``超参数:`` Batch Size = 16 (优化显存利用)，Learning Rate = 0.001，Adam 优化器，300 Epochs。

``损失函数:``$L1$ Loss (相比 $L2$ 更能减少生成图的模糊感,训练比较稳定,但生成图的细节会丢失更多)。

## 网络架构 (Network Architecture)

1. 基础结构：`采用 U-Net` 结构的对称全卷积网络。


2. 编码器 (Encoder)：4 层下采样。在本次实验中修改了原框架中的`ReLU`激活函数，使用 `LeakyReLU` 激活函数增加非线性表达能力，采用BatchNorm2d 稳定分布。

3. 解码器 (Decoder)：4 层转置卷积 `(ConvTranspose2d)`。

4. 核心改进：在解码器中添加了 `skip connection`，将编码器中的特征图与解码器中的特征图进行拼接，以保留更多的上下文信息。并引入 `torch.nn.functional`中的`F.interpolate` (双线性插值) 确保在特征融合时空间维度严格匹配。(否则会报`Runtime Error`,原因是代码框架是针对Facades数据集的，而Maps数据集会出现尺寸无法匹配的情况)。

## 实验结果

### 训练集效果展示(部分)

![image](https://github.com/tanhaotian2003/SA25001051_tanhaotian_DIP_2026Spring/blob/main/hw2/Pix2Pix/Result/train/result_5.png)
![image](https://github.com/tanhaotian2003/SA25001051_tanhaotian_DIP_2026Spring/blob/main/hw2/Pix2Pix/Result/train/result_4.png)

可以看出，模型在训练集上表现良好，Loss处于较低的水平(0.02)

### 测试集效果展示(部分)
![image](https://github.com/tanhaotian2003/SA25001051_tanhaotian_DIP_2026Spring/blob/main/hw2/Pix2Pix/Result/val/result_1.png)
![image](https://github.com/tanhaotian2003/SA25001051_tanhaotian_DIP_2026Spring/blob/main/hw2/Pix2Pix/Result/val/result_2.png)

可以看出，模型在测试集上也表现良好，Validation Loss较低（0.06）且对于epoch稳定，说明我们训练的模型有着较好的泛化能力。
## 实验总结 
本实验 验证了采用残差连接的 ``U-Net``网络架构 在保持图像结构信息，对图像进行语义划分方面的优越性。同时分析了不同batchsize对训练过程的影响。


