import torch
import torch.nn as nn
import DiceLoss as DiceLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, segmentation_loss_fn=nn.BCEWithLogitsLoss(), dice_weight=0.5):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.segmentation_loss_fn = segmentation_loss_fn
        self.dice_loss_fn = DiceLoss.DiceLoss()
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight  # 用于控制 Dice 损失和交叉熵损失的权重

    def forward(self, classification_output, segmentation_output, classification_target, segmentation_target):
        # 分类损失
        classification_loss = self.classification_loss_fn(classification_output, classification_target)

        # 计算 Dice 损失和交叉熵损失
        dice_loss = self.dice_loss_fn(segmentation_output, segmentation_target)
        segmentation_loss = self.segmentation_loss_fn(segmentation_output, segmentation_target)

        # 总分割损失（结合 Dice 损失和交叉熵损失）
        total_segmentation_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * segmentation_loss

        # 总损失
        total_loss = self.alpha * classification_loss + (1 - self.alpha) * total_segmentation_loss
        return total_loss


class MultiLoss(nn.Module):
    def __init__(self, segmentation_loss_fn=nn.BCEWithLogitsLoss(), dice_weight=0.5):
        super(MultiLoss, self).__init__()
        self.segmentation_loss_fn = segmentation_loss_fn
        self.dice_loss_fn = DiceLoss.DiceLoss()
        self.dice_weight = dice_weight  # 用于控制 Dice 损失和交叉熵损失的权重

    def forward(self, segmentation_output, segmentation_target):

        # 计算 Dice 损失和交叉熵损失
        dice_loss = self.dice_loss_fn(segmentation_output, segmentation_target)
        segmentation_loss = self.segmentation_loss_fn(segmentation_output, segmentation_target)

        # 总分割损失（结合 Dice 损失和交叉熵损失）
        total_segmentation_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * segmentation_loss

        return total_segmentation_loss