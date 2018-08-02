"""

共有92例可使用的数据

将这些数据随机分成80作为训练集，12例作为评价集
"""

import os
import random
import shutil

import numpy as np
import SimpleITK as sitk

mr_path = '/home/zcy/Desktop/data/mr/'
seg_path = '/home/zcy/Desktop/data/seg/'

val_mr_path = '/home/zcy/Desktop/val/mr/'
val_seg_path = '/home/zcy/Desktop/val/seg/'

num_val = 12

# 新产生的训练数据存储路径
if os.path.exists('/home/zcy/Desktop/val/') is True:
    shutil.rmtree('/home/zcy/Desktop/val/')
os.mkdir('/home/zcy/Desktop/val/')
os.mkdir(val_mr_path)
os.mkdir(val_seg_path)

id_list = []
for mr_file in os.listdir(mr_path):

    # 提取病例ID
    mr_id = mr_file.split('T1')[0]

    id_list.append(mr_id)


val_id_list = []
for _ in range(num_val):
    temp_id = random.choice(id_list)
    id_list.remove(temp_id)
    val_id_list.append(temp_id)

for index, item in enumerate(val_id_list, start=1):
    print(index, item)
print('--------------')


for file_index, val_id in enumerate(val_id_list):

    liver_seg = val_id + 'T1Liver.mha'
    kidneyleft_seg = val_id + 'T1KidneyLeft.mha'
    kidneyright_seg = val_id + 'T1KidneyRight.mha'

    mr_file = val_id + 'T1VenousPhase.mha'

    # 读取mr数据
    mr = sitk.ReadImage(os.path.join(mr_path, mr_file), sitk.sitkInt16)
    mr_array = sitk.GetArrayFromImage(mr)

    # 读取对应的金标准
    try:
        liver_seg = sitk.ReadImage(os.path.join(seg_path, liver_seg), sitk.sitkUInt8)
        liver_array = sitk.GetArrayFromImage(liver_seg)
        print('liver shape:', liver_array.shape)

    except Exception as e:
        print('error!')
        liver_array = np.zeros(mr_array.shape)

    try:
        kidneyleft_seg = sitk.ReadImage(os.path.join(seg_path, kidneyleft_seg), sitk.sitkUInt8)
        kidneyleft_array = sitk.GetArrayFromImage(kidneyleft_seg)
        print('kidney left shape', kidneyleft_array.shape)

    except Exception as e:
        print('error!')
        kidneyleft_array = np.zeros(mr_array.shape)

    try:
        kidneyright_seg = sitk.ReadImage(os.path.join(seg_path, kidneyright_seg), sitk.sitkUInt8)
        kidneyright_array = sitk.GetArrayFromImage(kidneyright_seg)
        print('kidney right shape', kidneyright_array.shape)

    except Exception as e:
        print('error!')
        kidneyright_array = np.zeros(mr_array.shape)

    # 将分散的金标准合并成一个
    seg_array = np.zeros(mr_array.shape)
    seg_array[liver_array == 1] = 1
    seg_array[kidneyleft_array == 1] = 2
    seg_array[kidneyright_array == 1] = 3
    seg_array = seg_array.astype(np.uint8)

    # 保存数据
    new_mr_array = mr_array
    new_seg_array = seg_array

    new_mr = sitk.GetImageFromArray(new_mr_array)

    new_mr.SetDirection(mr.GetDirection())
    new_mr.SetOrigin(mr.GetOrigin())
    new_mr.SetSpacing(mr.GetSpacing())

    new_seg = sitk.GetImageFromArray(new_seg_array)

    new_seg.SetDirection(mr.GetDirection())
    new_seg.SetOrigin(mr.GetOrigin())
    new_seg.SetSpacing(mr.GetSpacing())

    new_mr_name = 'img-' + str(file_index) + '.mha'
    new_seg_name = 'label-' + str(file_index) + '.mha'

    sitk.WriteImage(new_mr, os.path.join(val_mr_path, new_mr_name))
    sitk.WriteImage(new_seg, os.path.join(val_seg_path, new_seg_name))

    # 删除原始文件
    os.system('rm ' + mr_path + val_id + '*')
    os.system('rm ' + seg_path + val_id + '*')
