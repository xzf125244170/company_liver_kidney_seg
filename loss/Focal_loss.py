"""
focal loss
对交叉熵损失函数的扩展，更善于处理数据不平衡的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

num_organ = 3


class CELoss(nn.Module):
    def __init__(self, alpha=2):
        """
        :param alpha: focal loss中的指数项的次数
        """
        super().__init__()

        self.alpha = alpha
        self.loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, pred, target):
        """
        :param pred: (B, 4, 48, 384, 512)
        :param target: (B, 48, 384, 512)
        """

        # 计算正样本的数量，这里所谓正样本就是属于器官的体素的个数
        num_target = (target > 0).type(torch.cuda.FloatTensor).sum()

        target = target.type(torch.LongTensor).cuda()

        # 计算交叉熵损失值
        loss = self.loss(pred, target)

        # 将金标准转换为独热码
        one_hot_list = []
        for index in range(num_organ + 1):
            one_hot = torch.zeros(target.size())
            one_hot[target == index] = 1
            one_hot_list.append(one_hot)
        one_hot = torch.stack(one_hot_list, dim=1)

        one_hot = one_hot.cuda()
        pred = F.softmax(pred, dim=1)

        # 对已经可以良好分类的数据的损失值进行衰减
        exponential_term_stage1 = (1 - (pred * one_hot).max(dim=1)[0]) ** self.alpha

        loss *= exponential_term_stage1

        # 如果这一批数据中没有正样本，(虽然这样的概率非常小，但是还是要避免一下)
        if num_target == 0:
            # 则使用全部样本的数量进行归一化，和正常的CE损失一样
            loss = loss.mean()
        else:
            # 否侧用正样本的数量对损失值进行归一化
            loss = loss.sum() / num_target

        return loss