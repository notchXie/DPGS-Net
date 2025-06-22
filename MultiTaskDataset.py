import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd


class MultiTaskDataset(Dataset):
    def __init__(self, root_dir, fold_ids, csv_path):
        """
        初始化多任务数据集，支持加载指定的多个fold的数据，同时加载分类标签。

        :param root_dir: 数据集根目录
        :param fold_ids: 当前使用的fold ID列表，例如 [1, 2, 3, 4]
        :param csv_path: 包含分类标签的CSV文件路径，格式为 image_name,class_label
        """
        self.root_dir = root_dir
        self.fold_ids = fold_ids
        self.imgs = []
        self.masks = []
        self.imgs_per_fold = []
        self.imgs_per_fold_start_idx = []

        # 加载分类标签CSV
        self.class_labels = self._load_classification_labels(csv_path)

        # 获取所选折叠的数据
        for fold_id in fold_ids:
            fold_dir = os.path.join(root_dir, f"fold_{fold_id}")
            fold_images = os.listdir(os.path.join(fold_dir, 'p_image'))
            fold_masks = os.listdir(os.path.join(fold_dir, 'p_mask'))

            self.imgs.extend(fold_images)
            self.masks.extend(fold_masks)
            self.imgs_per_fold.append(len(fold_images))
            self.imgs_per_fold_start_idx.append(len(self.imgs) - len(fold_images))

        # 数据预处理
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def _load_classification_labels(self, csv_path):
        """
        加载分类标签CSV文件，生成字典。

        :param csv_path: CSV文件路径
        :return: 字典 {image_name: class_label}
        """
        df = pd.read_csv(csv_path)
        return {row["image_name"]: row["class_label"] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.imgs)

    def resize(self, image, mask):
        """
        调整图像和掩膜大小。
        """
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        return image, mask

    def __getitem__(self, idx):
        # 根据索引计算是哪个fold的数据
        fold_idx = 0
        for i, img_count in enumerate(self.imgs_per_fold):
            if idx < img_count:
                fold_idx = i
                break
            idx -= img_count

        # 计算当前fold在 self.imgs 中的起始索引
        fold_start_idx = self.imgs_per_fold_start_idx[fold_idx]
        local_idx = idx + fold_start_idx

        fold_id = self.fold_ids[fold_idx]
        fold_dir = os.path.join(self.root_dir, f"fold_{fold_id}")

        # 获取图像和掩膜路径
        img_path = os.path.join(fold_dir, 'p_image', self.imgs[local_idx])
        mask_path = os.path.join(fold_dir, 'p_mask', self.masks[local_idx])

        # 读取图像和掩膜
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        # 调整图像和掩膜大小
        img, mask = self.resize(img, mask)

        # 应用transform
        img = self.transform(img)
        mask = self.transform(mask)

        # 将mask转为二进制
        mask = torch.where(mask > 0, 1, 0).float()

        # 获取分类标签
        img_name = os.path.basename(img_path)  # 提取图像文件名
        class_label = self.class_labels.get(img_name, 0)  # 默认值-1表示未找到标签

        return img, mask, class_label
