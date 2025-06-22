import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        初始化 Dice Loss.

        :param smooth: 避免分母为 0 的平滑项.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        计算 Dice Loss.

        :param inputs: 预测结果, 形状为 (N, C, H, W)
        :param targets: 真实标签, 形状为 (N, C, H, W)
        :return: Dice 损失值.
        """
        # 使用 sigmoid 将预测结果转换到 [0, 1]
        inputs = torch.sigmoid(inputs)

        # 展平
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Dice 系数计算
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 取平均值作为损失
        dice_loss = 1 - dice.mean()
        return dice_loss
