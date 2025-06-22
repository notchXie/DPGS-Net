import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import module.pre_encoder as pre_encoder

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载模型
model = pre_encoder.Unet(in_channels=3, out_channels=1)
# 修改加载方式以解决警告
model.load_state_dict(torch.load("endunet_15.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()  # 使用eval模式

# 2. 定义Grad-CAM类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def forward(self, x):
        return self.model(x)
    
    def __call__(self, x, class_idx=0):
        # 前向传播
        output = self.forward(x)
        
        # 创建目标(对于分割任务，我们关注特定类别的输出)
        target = torch.zeros_like(output)
        target[:, class_idx, :, :] = 1
        
        # 反向传播
        self.model.zero_grad()
        torch.sum(output * target).backward(retain_graph=True)
        
        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 加权激活图
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)  # 只保留正影响
        heatmap /= torch.max(heatmap)  # 归一化
        
        return heatmap.cpu().numpy(), output

# 3. 选择目标层 - 修改为正确的层选择方式
# 我们需要找到最后一个下采样层的最后一个卷积层
# 查看你的MFDCMmodule结构，选择适当的层
# 这里假设MFDCMmodule有一个convFusion属性
target_layer = model.encoder4.convFusion[-2]  # 选择convFusion中的倒数第二层(通常是最后一个卷积)

# 4. 创建Grad-CAM实例
grad_cam = GradCAM(model, target_layer)

# 5. 准备输入图像
def preprocess_image(img_path):
    # 读取和预处理图像
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 根据你的模型输入尺寸调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

# 6. 可视化函数
def visualize_cam(image, heatmap, output, alpha=0.5):
    # 转换为numpy数组
    image = np.array(image)
    image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    
    # 归一化热力图
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = np.uint8(255 * heatmap)
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 叠加热力图和原图
    superimposed_img = heatmap_colored * alpha + image * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # 创建可视化图
    plt.figure(figsize=(15, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # 热力图
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title('Superimposed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('unet_metrics.png')
    plt.show()
    
    # 返回模型输出(用于分割任务)
    output = output.squeeze().detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(output, cmap='gray')
    plt.title('Model Output')
    plt.axis('off')
    plt.show()

# 7. 运行Grad-CAM并可视化
def run_grad_cam(img_path):
    # 预处理图像
    img_tensor, original_img = preprocess_image(img_path)
    
    # 获取热力图和模型输出
    heatmap, output = grad_cam(img_tensor)
    
    # 可视化结果
    visualize_cam(original_img, heatmap, output)

# 8. 测试示例 (替换为你的图像路径)
img_path = "20197.PNG"  # 替换为你的测试图像路径
run_grad_cam(img_path)