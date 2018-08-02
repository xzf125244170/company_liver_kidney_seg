"""

随机取样
"""

import os
import random

import torch
from torch.utils.data import Dataset as dataset

import SimpleITK as sitk

on_server = True
size = 48


class Dataset(dataset):
    def __init__(self, mr_dir, seg_dir):
        """

        :param mr_dir: mr数据的地址
        :param seg_dir: 金标准的地址
        """

        self.mr_list = os.listdir(mr_dir)
        self.seg_list = list(map(lambda x: x.replace('img', 'label'), self.mr_list))

        self.mr_list = list(map(lambda x: os.path.join(mr_dir, x), self.mr_list))
        self.seg_list = list(map(lambda x: os.path.join(seg_dir, x), self.seg_list))

    def __getitem__(self, index):
        """

        :return: torch.Size([B, 1, 48, 256, 256]) torch.Size([B, 48, 384, 512])
        """

        mr_path = self.mr_list[index]
        seg_path = self.seg_list[index]

        # 将CT和金标准读入到内存中
        mr = sitk.ReadImage(mr_path, sitk.sitkInt16)
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)

        mr_array = sitk.GetArrayFromImage(mr)
        seg_array = sitk.GetArrayFromImage(seg)

        # 在slice平面内随机选取48张slice
        start_slice = random.randint(0, mr_array.shape[0] - size)
        end_slice = start_slice + size - 1

        mr_array = mr_array[start_slice:end_slice + 1, :, :]
        seg_array = seg_array[start_slice:end_slice + 1, :, :]

        # 处理完毕，将array转换为tensor
        mr_array = torch.FloatTensor(mr_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array)

        return mr_array, seg_array

    def __len__(self):

        return len(self.mr_list)


mr_dir = '/home/zcy/Desktop/train/mr/' \
    if on_server is False else './train/mr/'
seg_dir = '/home/zcy/Desktop/train/seg/' \
    if on_server is False else './train/seg/'

train_ds = Dataset(mr_dir, seg_dir)


# # 测试代码
# from torch.utils.data import DataLoader
# train_dl = DataLoader(train_ds, 6, True, num_workers=2)
# for index, (mr, seg) in enumerate(train_dl):
#
#     print(index, mr.size(), seg.size())
#     print('----------------')
