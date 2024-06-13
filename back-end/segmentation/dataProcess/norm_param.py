import os
import numpy as np
from tqdm import trange

path = '/workplace/dataset/MODIS_new/train_seg1/'
file_all = os.listdir(path)

min_list = [float('inf')] * 9
max_list = [-float('inf')] * 9

print('begin positive')
for j in trange(len(file_all)):
    data = np.load(path + str(j + 1) + '.npy')
    for i in range(9):
        # print(f'\nchannel**{i}**begins in positive dataset')
        min_list[i] = min(min_list[i], np.min(data[:, i]))
        max_list[i] = max(max_list[i], np.max(data[:, i]))


for i in range(9):
    print(f'min of channel{i}')
    print(min_list[i])
    print(f'max of channel{i}')
    print(max_list[i])