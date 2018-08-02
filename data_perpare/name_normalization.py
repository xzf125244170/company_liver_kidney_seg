"""

将金标准数据的命名规范化方便之后的读取
"""

import os

import SimpleITK as sitk


mr_path = '/home/zcy/Desktop/data/mr/'
seg_path = '/home/zcy/Desktop/data/seg/'


for mr_file in os.listdir(mr_path):

    # 将mr数据读取到内存中
    mr = sitk.ReadImage(os.path.join(mr_path, mr_file), sitk.sitkInt16)
    mr_array = sitk.GetArrayFromImage(mr)

    # 提取病例ID
    mr_id = mr_file.split('T1')[0]

    # 将mr数据的命名规范化
    os.system('mv ' + mr_path + mr_id + '* ' + mr_path + mr_id + 'T1VenousPhase.mha')

    # 将金标准数据的命名规范化
    os.system('mv ' + seg_path + mr_id + '*Liver* ' + seg_path + mr_id + 'T1Liver.mha')
    os.system('mv ' + seg_path + mr_id + '*Left* ' + seg_path + mr_id + 'T1KidneyLeft.mha')
    os.system('mv ' + seg_path + mr_id + '*Right* ' + seg_path + mr_id + 'T1KidneyRight.mha')
