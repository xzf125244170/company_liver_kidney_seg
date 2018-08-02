"""

纯测试脚本

共3种器官＋背景
(0) 背景
(1) 肝脏
(2) 左肾
(3) 右肾
"""

import os
from time import time

import torch
import torch.nn.functional as F

import numpy as np
import SimpleITK as sitk

from net.ResUNet import ResUNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

val_mr_dir = './val/mr/'
file = 'img-0.mha'
organ_pred_dir = './val/pred/'
module_dir = './module/.pth'

size = 48

# 定义网络并加载参数
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()

# 开始正式进行测试
start_time = time()

# 将CT读入内存
mr = sitk.ReadImage(os.path.join(val_mr_dir, file), sitk.sitkInt16)
mr_array = sitk.GetArrayFromImage(mr)
origin_shape = mr_array.shape

depth = mr_array.shape[0]

# 对CT使用双三次算法进行插值，插值之后的array依然是int16
mr_array = torch.FloatTensor(mr_array).unsqueeze(dim=0).unsqueeze(dim=0)
mr_array = F.upsample(mr_array, (depth, 256, 256), mode='trilinear').squeeze().detach().numpy()
mr_array = np.round(mr_array).astype(np.int16)

# 在轴向上进行切块取样
flag = False
start_slice = 0
end_slice = start_slice + size - 1
mr_array_list = []

while end_slice <= mr_array.shape[0] - 1:
    mr_array_list.append(mr_array[start_slice:end_slice + 1, :, :])

    start_slice = end_slice + 1
    end_slice = start_slice + size - 1

# 当无法整除的时候反向取最后一个block
if end_slice is not mr_array.shape[0] - 1:
    flag = True
    count = mr_array.shape[0] - start_slice
    mr_array_list.append(mr_array[-size:, :, :])

outputs_list = []
with torch.no_grad():
    for mr_array in mr_array_list:

        mr_tensor = torch.FloatTensor(mr_array).cuda()
        mr_tensor = mr_tensor.unsqueeze(dim=0)
        mr_tensor = mr_tensor.unsqueeze(dim=0)

        outputs = net(mr_tensor)

        # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
        outputs_list.append(outputs.cpu().detach().numpy())
        del outputs

# 执行完之后开始拼接结果
pred_seg = np.concatenate(outputs_list[0:-1], axis=2)

if flag is False:
    pred_seg = np.concatenate([pred_seg, outputs_list[-1]], axis=2)
else:
    pred_seg = np.concatenate([pred_seg, outputs_list[-1][:, :,  -count:, :, :]], axis=2)

# 使用线性插值将预测的分割结果缩放到原始nii大小
pred_seg = torch.FloatTensor(pred_seg)
pred_seg = F.upsample(pred_seg, origin_shape, mode='trilinear').squeeze().detach().numpy()
pred_seg = np.argmax(pred_seg, axis=0)
pred_seg = np.round(pred_seg).astype(np.uint8)

print('size of pred: ', pred_seg.shape)
print('size of GT: ', origin_shape)

# 将预测的结果保存为nii数据
pred_seg = sitk.GetImageFromArray(pred_seg)

pred_seg.SetDirection(mr.GetDirection())
pred_seg.SetOrigin(mr.GetOrigin())
pred_seg.SetSpacing(mr.GetSpacing())

sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file.replace('img', 'organ')))

speed = time() - start_time

print('this case use {:.3f} s'.format(speed))
print('-----------------------')
