import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms


# --------------------- 1. 自定义数据集 ---------------------
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file, header=None, names=['image_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------- 2. 数据预处理与加载 ---------------------
def load_data(csv_file, img_dir, batch_size=16):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # ResNet输入固定尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取CSV文件并划分训练集和测试集
    data = pd.read_csv(csv_file, header=None, names=['image_name', 'label'])
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # 保存划分后的CSV文件（可选）
    train_csv = 'train.csv'
    test_csv = 'test.csv'
    train_data.to_csv(train_csv, index=False, header=False)
    test_data.to_csv(test_csv, index=False, header=False)

    # 创建Dataset和DataLoader
    train_dataset = ImageDataset(train_csv, img_dir, transform)
    test_dataset = ImageDataset(test_csv, img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# --------------------- 3. 模型定义 ---------------------
def build_model(num_classes):
    model = models.resnet101(pretrained=True)
    # 替换最后的全连接层以适配分类任务
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --------------------- 4. 训练与测试函数 ---------------------
def train_model(model, train_loader, test_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 测试模型
        test_model(model, test_loader, device)


def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# --------------------- 5. 主函数 ---------------------
if __name__ == "__main__":
    csv_file = r'D:\GitData\Unet\dataset\output\image_labels.csv'
    img_dir = r'D:\GitData\Datasets\Thyroid Dataset\DDTI dataset\DDTI\2_preprocessed_data\stage1\p_image'
    batch_size = 24
    num_classes = 3
    num_epochs = 30

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    train_loader, test_loader = load_data(csv_file, img_dir, batch_size)

    # 构建模型
    model = build_model(num_classes)

    # 训练模型
    train_model(model, train_loader, test_loader, device, num_epochs)
