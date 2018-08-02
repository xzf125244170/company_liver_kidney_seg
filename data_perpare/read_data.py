"""

读取原始mr数据，分析数据情况，确保数据正确
"""

import os

import SimpleITK as sitk


mr_path = '/home/zcy/Desktop/data/mr/'
seg_path = '/home/zcy/Desktop/data/seg/'

shape1_list = []

for mr_file in os.listdir(mr_path):

    print('file name:', mr_file)

    try:
        mr = sitk.ReadImage(os.path.join(mr_path, mr_file), sitk.sitkInt16)
        mr_array = sitk.GetArrayFromImage(mr)

    except Exception as e:
        print(mr_file, 'error!')

    print(mr_array.shape)
    print(mr.GetSpacing())

    if mr_array.shape[0] != 88:
        print('shape 0 outlier')
    if mr_array.shape[2] != 512:
        print('shape 2 outlier')

    shape1_list.append(mr_array.shape[1])

shape1_list = set(shape1_list)
print(shape1_list)
