import os
import numpy as np
from tqdm import trange

path = '/workplace/dataset/MODIS_new/train_80/'
positive_all = os.listdir(path + 'positive')
negative_all = os.listdir(path + 'negative')

min_pos_list = [float('inf')] * 9
max_pos_list = [-float('inf')] * 9
min_neg_list = [float('inf')] * 9
max_neg_list = [-float('inf')] * 9

print('begin positive')
for j in trange(len(positive_all)):
    pos_data = np.load('/workplace/dataset/MODIS_new/train_80/positive/' + str(j + 1) + '.npy')
    for i in range(9):
        # print(f'\nchannel**{i}**begins in positive dataset')
        min_pos_list[i] = min(min_pos_list[i], np.min(pos_data[:, i]))
        max_pos_list[i] = max(max_pos_list[i], np.max(pos_data[:, i]))

print('begin negative')
for k in trange(len(negative_all)):
    neg_data = np.load('/workplace/dataset/MODIS_new/train_80/negative/' + str(k + 1) + '.npy')
    for i in range(9):
        # print(f'\nchannel**{i}**begins in negative dataset')
        min_neg_list[i] = min(min_neg_list[i], np.min(neg_data[:, i]))
        max_neg_list[i] = max(max_neg_list[i], np.max(neg_data[:, i]))

for i in range(9):
    min_final = min(min_pos_list[i], min_neg_list[i])
    max_final = max(max_pos_list[i], max_neg_list[i])
    print(f'min of channel{i}')
    print(min_final)
    print(f'max of channel{i}')
    print(max_final)