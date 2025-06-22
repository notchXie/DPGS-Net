import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class datasets(Dataset):
    def __init__(self, root_dir, fold_ids):
        """
        初始化数据集，支持加载指定的多个fold的数据作为训练集或测试集。

        :param root_dir: 数据集根目录
        :param fold_ids: 当前使用的fold ID列表，例如 [1, 2, 3, 4] 表示加载 fold_1, fold_2, fold_3, fold_4 的数据
        """
        self.root_dir = root_dir
        self.fold_ids = fold_ids  # 当前使用的fold ID列表
        self.imgs = []  # 存储所有折叠的图像
        self.masks = []  # 存储所有折叠的掩膜
        self.imgs_per_fold = []  # 每个fold的数据数量
        self.imgs_per_fold_start_idx = []  # 存储每个fold在 self.imgs 中的起始索引

        # 获取所选折叠的数据
        for fold_id in fold_ids:
            fold_dir = os.path.join(root_dir, f"fold_{fold_id}")
            fold_images = os.listdir(os.path.join(fold_dir, 'p_image'))
            fold_masks = os.listdir(os.path.join(fold_dir, 'p_mask'))

            self.imgs.extend(fold_images)
            self.masks.extend(fold_masks)
            self.imgs_per_fold.append(len(fold_images))  # 保存每个fold的图片数量
            self.imgs_per_fold_start_idx.append(len(self.imgs) - len(fold_images))  # 保存当前fold起始索引

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.imgs)

    def resize(self, image, mask):
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
            idx -= img_count  # 调整idx为当前折叠的数据索引

        # 计算当前折叠在 self.imgs 中的索引范围
        fold_start_idx = self.imgs_per_fold_start_idx[fold_idx]
        local_idx = idx + fold_start_idx  # 当前折叠的局部索引

        fold_id = self.fold_ids[fold_idx]  # 当前fold的ID
        fold_dir = os.path.join(self.root_dir, f"fold_{fold_id}")

        img_path = os.path.join(fold_dir, 'p_image', self.imgs[local_idx])
        mask_path = os.path.join(fold_dir, 'p_mask', self.masks[local_idx])

        # 读取图像和掩膜
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # 读取为单通道

        img, mask = self.resize(img, mask)

        img = self.transform(img)
        mask = self.transform(mask)

        mask = torch.where(mask > 0, 1, 0)
        mask = mask.float()

        return img, mask
