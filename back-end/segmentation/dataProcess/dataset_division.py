# 导入需要的模块
import os
import numpy as np
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

pos_path = "/workplace/dataset/MODIS_new/train_80/positive" # train_80源文件夹，包含4843个npy文件
# train:valid:test=6:2:2
# 训练集
train_dataset_path = "/workplace/dataset/MODIS_new/train_seg1"
if not os.path.exists(train_dataset_path):
    os.makedirs(train_dataset_path)
# 验证集
valid_dataset_path = "/workplace/dataset/MODIS_new/valid_seg1"
if not os.path.exists(valid_dataset_path):
    os.makedirs(valid_dataset_path)
# 测试集
test_dataset_path = "/workplace/dataset/MODIS_new/test_seg1"
if not os.path.exists(test_dataset_path):
    os.makedirs(test_dataset_path)


# 获取源文件夹中的所有npy文件的文件名
file_list = os.listdir(pos_path)
cnt_pos, cnt_train, cnt_valid, cnt_test=0, 0, 0, 0

for file in file_list:
    cnt_pos += 1
    print(file)
    img = np.load(os.path.join(pos_path, file))

    # 训练集
    if cnt_pos <= 2095:
        cnt_train += 1
        train_save = os.path.join(train_dataset_path, str(cnt_train)+'.npy')
        np.save(train_save, img)
    # 验证集
    elif cnt_pos % 2 == 0:
        cnt_valid += 1
        valid_save = os.path.join(valid_dataset_path, str(cnt_valid)+'.npy')
        np.save(valid_save, img)
    # 测试集
    else:
        cnt_test += 1
        test_save = os.path.join(test_dataset_path, str(cnt_test)+'.npy')
        np.save(test_save, img)

file_train = os.listdir(train_dataset_path)
print(len(file_train))
file_valid = os.listdir(valid_dataset_path)
print(len(file_valid))
file_test = os.listdir(test_dataset_path)
print(len(file_test))

# # 将文件名列表随机划分为三个子列表，比例为6:2:2
# # 设置随机种子
# seed = 42
# train_list, valid_test_list = train_test_split(file_list, test_size=0.4, random_state=seed)
# valid_list, test_list = train_test_split(valid_test_list, test_size=0.5, random_state=seed)

# cnt_train, cnt_valid, cnt_test=0, 0, 0
# # 将每个子列表中的文件复制到对应的目标文件夹中
# for file in train_list:
#     cnt_train+=1
#     new_file_name = str(cnt_train)+'.npy'

#     old_path = os.path.join(pos_path, file)
#     new_path = os.path.join(train_dataset_path, file)
#     shutil.copy(old_path, train_dataset_path)

#     ## 将文件复制过去后重命名
#     os.rename(new_path, os.path.join(train_dataset_path, new_file_name))
#     print(cnt_train)
    
# for file in valid_list:
#     cnt_valid+=1
#     new_file_name = str(cnt_valid)+'.npy'

#     old_path = os.path.join(pos_path, file)
#     new_path = os.path.join(valid_dataset_path, file)
#     shutil.copy(os.path.join(pos_path, file), valid_dataset_path)
#     os.rename(new_path, os.path.join(valid_dataset_path, new_file_name))
#     print(cnt_valid)

# for file in test_list:
#     cnt_test+=1
#     new_file_name = str(cnt_test)+'.npy'

#     old_path = os.path.join(pos_path, file)
#     new_path = os.path.join(test_dataset_path, file)
#     shutil.copy(os.path.join(pos_path, file), test_dataset_path)
#     os.rename(new_path, os.path.join(test_dataset_path, new_file_name))
#     print(cnt_test)

