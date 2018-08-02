from time import time
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from loss.Dice_loss import DiceLoss
from net.ResUNet import net
from dataset.dataset import train_ds


# 定义超参数
on_server = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
cudnn.benchmark = True
Epoch = 2000
leaing_rate_base = 1e-4

batch_size = 1 if on_server is False else 3
num_workers = 1 if on_server is False else 3
pin_memory = False if on_server is False else True

net = torch.nn.DataParallel(net).cuda()
net.train()

# 定义数据加载
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 定义损失函数
loss_func = DiceLoss()

# 定义优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate_base)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [1000])

# 训练网络
start = time()
for epoch in range(Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        outputs = net(ct)

        loss = loss_func(outputs, seg)

        with torch.no_grad():
            mean_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 is 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))

    with torch.no_grad():
        mean_loss = sum(mean_loss) / len(mean_loss)

    if epoch % 10 is 0:
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))
