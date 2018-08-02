"""
在12例随机挑选的数据上做测试
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
import xlsxwriter as xw

from net.ResUNet import ResUNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

val_mr_dir = './val/mr/'
val_seg_dir = './val/seg/'

organ_pred_dir = './val/pred/'

module_dir = './module/net360-0.071-0.230.pth'

size = 48


organ_list = [
    'liver',
    'left kidney',
    'right kidney',
]

# 创建一个表格对象，并添加一个sheet，后期配合window的excel来出图
workbook = xw.Workbook('./result.xlsx')
worksheet = workbook.add_worksheet('result')

# 设置单元格格式
bold = workbook.add_format()
bold.set_bold()

center = workbook.add_format()
center.set_align('center')

center_bold = workbook.add_format()
center_bold.set_bold()
center_bold.set_align('center')

worksheet.set_column(1, len(os.listdir(val_mr_dir)), width=15)
worksheet.set_column(0, 0, width=30, cell_format=center_bold)
worksheet.set_row(0, 20, center_bold)

# 写入文件名称
worksheet.write(0, 0, 'file name')
for index, file_name in enumerate(os.listdir(val_mr_dir), start=1):
    worksheet.write(0, index, file_name)

# 写入各项评价指标名称
for index, organ_name in enumerate(organ_list, start=1):
    worksheet.write(index, 0, organ_name)
worksheet.write(4, 0, 'speed')
worksheet.write(5, 0, 'shape')


# 定义网络并加载参数
net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()


# 开始正式进行测试
for file_index, file in enumerate(os.listdir(val_mr_dir)):

    start_time = time()

    # 将CT读入内存
    mr = sitk.ReadImage(os.path.join(val_mr_dir, file), sitk.sitkInt16)
    mr_array = sitk.GetArrayFromImage(mr)

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

    # 将金标准读入内存来计算dice系数
    seg = sitk.ReadImage(os.path.join(val_seg_dir, file.replace('img', 'label')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # 使用线性插值将预测的分割结果缩放到原始nii大小
    pred_seg = torch.FloatTensor(pred_seg)
    pred_seg = F.upsample(pred_seg, seg_array.shape, mode='trilinear').squeeze().detach().numpy()
    pred_seg = np.argmax(pred_seg, axis=0)
    pred_seg = np.round(pred_seg).astype(np.uint8)

    print('size of pred: ', pred_seg.shape)
    print('size of GT: ', seg_array.shape)

    worksheet.write(5, file_index + 1, str(seg_array.shape))

    # 计算每一种器官的dice系数，并将结果写入表格中存储
    for organ_index, organ in enumerate(organ_list, start=1):

        pred_organ = np.zeros(pred_seg.shape)
        target_organ = np.zeros(seg_array.shape)

        pred_organ[pred_seg == organ_index] = 1
        target_organ[seg_array == organ_index] = 1

        # 如果该例数据中不存在某一种器官，在表格中记录 None 跳过即可
        if target_organ.sum() == 0:
            worksheet.write(organ_index, file_index + 1, 'None')

        else:
            dice = (2 * pred_organ * target_organ).sum() / (pred_organ.sum() + target_organ.sum())
            worksheet.write(organ_index, file_index + 1, dice)

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(pred_seg)

    pred_seg.SetDirection(mr.GetDirection())
    pred_seg.SetOrigin(mr.GetOrigin())
    pred_seg.SetSpacing(mr.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(organ_pred_dir, file.replace('img', 'organ')))
    del pred_seg

    speed = time() - start_time

    worksheet.write(4, file_index + 1, speed)

    print('this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 最后安全关闭表格
workbook.close()
