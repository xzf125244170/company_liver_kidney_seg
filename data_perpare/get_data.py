"""

将分散的原始数据整理到相应的文件夹中
拼写错误等在所难免，完成之后还是需要检查一下
"""

import os
import shutil

raw_data_dir = '/home/zcy/Desktop/raw_data/'

mr_dir = '/home/zcy/Desktop/data/mr/'
seg_dir = '/home/zcy/Desktop/data/seg/'

if os.path.exists('/home/zcy/Desktop/data/'):
    shutil.rmtree('/home/zcy/Desktop/data/')

os.mkdir('/home/zcy/Desktop/data')
os.mkdir(mr_dir)
os.mkdir(seg_dir)

for name in os.listdir(raw_data_dir):
    for folder in os.listdir(os.path.join(raw_data_dir, name)):
        if 'T1VenousPhase' not in folder:
            continue
        for file in os.listdir(os.path.join(raw_data_dir, name, folder)):

            os.system('mv ' + os.path.join(raw_data_dir, name, folder) + '/' + '*VenousPhase*.mha* ' + mr_dir)
            os.system('mv ' + os.path.join(raw_data_dir, name, folder) + '/' + '*Liver* ' + seg_dir)
            os.system('mv ' + os.path.join(raw_data_dir, name, folder) + '/' + '*Left* ' + seg_dir)
            os.system('mv ' + os.path.join(raw_data_dir, name, folder) + '/' + '*Right* ' + seg_dir)

        os.system('rmdir ' + os.path.join(raw_data_dir, name, folder))
    os.system('rmdir ' + os.path.join(raw_data_dir, name))
