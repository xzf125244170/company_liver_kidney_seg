"""
加权交叉熵损失函数
"""

import torch
import torch.nn as nn

num_organ = 3


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

        weight = torch.FloatTensor((1, 2, 4, 4)).cuda()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        """
        :param pred: (B, 4, 48, 384, 512)
        :param target: (B, 48, 384, 512)
        """
        target = target.type(torch.LongTensor).cuda()
        # 计算交叉熵损失值
        loss = self.loss(pred, target)

        return loss