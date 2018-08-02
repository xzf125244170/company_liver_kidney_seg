"""

将原始数据变成训练数据

背景：０
肝脏：１
左肾：２
右肾：３
"""

import os
import shutil
from time import time

import numpy as np
import SimpleITK as sitk

import torch
import torch.nn.functional as F

mr_path = '/home/zcy/Desktop/data/mr/'
seg_path = '/home/zcy/Desktop/data/seg/'

new_mr_path = '/home/zcy/Desktop/train/mr/'
new_seg_path = '/home/zcy/Desktop/train/seg/'

# 新产生的训练数据存储路径
if os.path.exists('/home/zcy/Desktop/train/') is True:
    shutil.rmtree('/home/zcy/Desktop/train/')
os.mkdir('/home/zcy/Desktop/train/')
os.mkdir(new_mr_path)
os.mkdir(new_seg_path)

file_index = 0  # 用于记录新产生的数据的序号
start_time = time()

for mr_file in os.listdir(mr_path):

    # 将mr数据读取到内存中
    mr = sitk.ReadImage(os.path.join(mr_path, mr_file), sitk.sitkInt16)
    mr_array = sitk.GetArrayFromImage(mr)

    depth = mr_array.shape[0]
    x = mr_array.shape[1]
    y = mr_array.shape[2]

    # 提取病例ID
    mr_id = mr_file.split('T1')[0]

    # 读取对应的金标准
    liver_seg = mr_id + 'T1Liver.mha'
    kidneyleft_seg = mr_id + 'T1KidneyLeft.mha'
    kidneyright_seg = mr_id + 'T1KidneyRight.mha'

    try:
        liver_seg = sitk.ReadImage(os.path.join(seg_path, liver_seg), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver_seg)
        print('liver shape:', liver_array.shape)

    except Exception as e:
        print(mr_file, 'error!')
        liver_array = np.zeros(mr_array.shape)
    try:
        kidneyleft_seg = sitk.ReadImage(os.path.join(seg_path, kidneyleft_seg), sitk.sitkUInt8)
        kidneyleft_array = sitk.GetArrayFromImage(kidneyleft_seg)
        print('kidney left shape', kidneyleft_array.shape)

    except Exception as e:
        print(mr_file, 'error!')
        kidneyleft_array = np.zeros(mr_array.shape)

    try:
        kidneyright_seg = sitk.ReadImage(os.path.join(seg_path, kidneyright_seg), sitk.sitkUInt8)
        kidneyright_array = sitk.GetArrayFromImage(kidneyright_seg)
        print('kidney right shape', kidneyright_array.shape)

    except Exception as e:
        print(mr_file, 'error!')
        kidneyright_array = np.zeros(mr_array.shape)

    print('file name:', mr_file)
    print('shape:', mr_array.shape)

    liver_array = torch.FloatTensor(liver_array).unsqueeze(dim=0).unsqueeze(dim=0)
    liver_array = F.upsample(liver_array, (depth, 384, 512), mode='trilinear').squeeze().detach().numpy()
    liver_array = np.round(liver_array).astype(np.int8)

    kidneyright_array = torch.FloatTensor(kidneyright_array).unsqueeze(dim=0).unsqueeze(dim=0)
    kidneyright_array = F.upsample(kidneyright_array, (depth, 384, 512), mode='trilinear').squeeze().detach().numpy()
    kidneyright_array = np.round(kidneyright_array).astype(np.int8)

    kidneyleft_array = torch.FloatTensor(kidneyleft_array).unsqueeze(dim=0).unsqueeze(dim=0)
    kidneyleft_array = F.upsample(kidneyleft_array, (depth, 384, 512), mode='trilinear').squeeze().detach().numpy()
    kidneyleft_array = np.round(kidneyleft_array).astype(np.int8)

    # 将分散的金标准合并成一个
    seg_array = np.zeros((depth, 384, 512))
    seg_array[liver_array == 1] = 1
    seg_array[kidneyleft_array == 1] = 2
    seg_array[kidneyright_array == 1] = 3
    seg_array = seg_array.astype(np.int8)

    mr_array = torch.FloatTensor(mr_array).unsqueeze(dim=0).unsqueeze(dim=0)
    mr_array = F.upsample(mr_array, (depth, 256, 256), mode='trilinear').squeeze().detach().numpy()
    mr_array = np.round(mr_array).astype(np.int16)

    # 保存数据
    new_mr_array = mr_array
    new_seg_array = seg_array

    new_mr = sitk.GetImageFromArray(new_mr_array)

    new_mr.SetDirection(mr.GetDirection())
    new_mr.SetOrigin(mr.GetOrigin())
    new_mr.SetSpacing(
        (mr.GetSpacing()[0] * (y / 256), mr.GetSpacing()[1] * (x / 256), mr.GetSpacing()[2]))

    new_seg = sitk.GetImageFromArray(new_seg_array)

    new_seg.SetDirection(mr.GetDirection())
    new_seg.SetOrigin(mr.GetOrigin())
    new_seg.SetSpacing(
        (mr.GetSpacing()[0] * (y / 384), mr.GetSpacing()[1] * (x / 512), mr.GetSpacing()[2]))

    new_mr_name = 'img-' + str(file_index) + '.mha'
    new_seg_name = 'label-' + str(file_index) + '.mha'

    sitk.WriteImage(new_mr, os.path.join(new_mr_path, new_mr_name))
    sitk.WriteImage(new_seg, os.path.join(new_seg_path, new_seg_name))

    # 每处理完一个数据，打印一次已经使用的时间
    print('already use {:.3f} min'.format((time() - start_time) / 60))
    print('-----------')

    file_index += 1

