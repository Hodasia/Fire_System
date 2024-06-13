import numpy as np
import os
import random
import sys,os

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import utils.data_config as data_config
dc=data_config.DataConfig

##采样训练集
def sample_train(batch_num, cut_batch):
    batch_data=[]
    path = '/workplace/dataset/MODIS_new/train_seg1/'

    ## sample num < img.shape[0] (64)
    if cut_batch:
        id=random.randint(1, dc.train_num)
        data=np.load(path+str(id)+'.npy')#64*11*15*15
        idx = random.sample(range(data.shape[0]), batch_num)
        sample_pos_data = np.take(data, idx, axis=0)
        batch_data.extend(sample_pos_data)
    else:
        ## sample num >= img.shape[0] (64)
        for _ in range(batch_num):
            id=random.randint(1, dc.train_num)
            data=np.load(path+str(id)+'.npy')#64*11*15*15
            batch_data.extend(data)

    batch_data=np.asarray(batch_data)

    batch_x=batch_data[:,:9]#256*9*15*15
    batch_y=batch_data[:,-1]#256*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y = np.expand_dims(batch_y, axis=1)

    return batch_x,batch_y

##采样验证集或测试集
def sample_val_test(idx_list, data_type):
    batch_data = []

    for idx in idx_list:
        data = np.load(os.path.join('/workplace/dataset/MODIS_new/', data_type+'_seg1/'+str(idx)+'.npy'))#(64,11,15,15)
        batch_data.extend(data)

    batch_data = np.asarray(batch_data)
    batch_x=batch_data[:,:9]#12150*9*15*15
    batch_y=batch_data[:,-1]#12150*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y = np.expand_dims(batch_y, axis=1)

    return batch_x,batch_y

# def sample_val_test(img_id, type):
#     data = np.load(os.path.join('/workplace/dataset/MODIS_new/', type+'_seg1/'+str(img_id)+'.npy'))#11*2030*1354

#     batch_x=data[:,:9]#12150*9*15*15
#     batch_y=data[:,-1]#12150*1*15*15
#     batch_y[batch_y<80]=0
#     batch_y[batch_y>=80]=1
#     batch_y = np.expand_dims(batch_y, axis=1)

#     return batch_x,batch_y