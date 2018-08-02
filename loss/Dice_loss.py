"""
多分类Dice loss
"""

import torch
import torch.nn as nn

num_organ = 3


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        :param pred: 经过放大之后(B, 4, 48, 384, 512)
        :param target: (B, 48, 384, 512)
        :return: Dice距离
        """

        # 首先将金标准拆开
        organ_target = torch.zeros((target.size(0), num_organ, 48, 384, 512))

        for organ_index in range(1, num_organ + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == organ_index] = 1
            organ_target[:, organ_index - 1, :, :, :] = temp_target
            # organ_target: (B, 3, 48, 128, 128)

        organ_target = organ_target.cuda()

        # 计算loss
        dice = 0.0

        for organ_index in range(1, num_organ + 1):
            dice += 2 * (pred[:, organ_index, :, :, :] * organ_target[:, organ_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(
                dim=1) / (pred[:, organ_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                          organ_target[:, organ_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

        dice /= num_organ

        # 返回的是dice距离
        return (1 - dice).mean()
