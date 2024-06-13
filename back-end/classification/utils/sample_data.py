import numpy as np
import os
import random
import sys,os

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import utils.data_config as data_config
dc=data_config.DataConfig

##采样训练集
def sample_train(batch_num_pos, batch_num_neg, cut_batch):
    batch_data=[]
    path_pos = '/workplace/dataset/MODIS_new/train_80/positive/'
    path_neg = '/workplace/dataset/MODIS_new/train_80/negative/'

    ## sample num < img.shape[0] (64)
    if cut_batch:
        pos_id=random.randint(1, dc.train_pos_num)
        pos_data=np.load(path_pos+str(pos_id)+'.npy')#32*11*224*224
        idx1 = random.sample(range(pos_data.shape[0]), batch_num_pos)
        sample_pos_data = np.take(pos_data, idx1, axis=0)
        batch_data.extend(sample_pos_data)

        neg_id=random.randint(1, dc.train_neg_num)
        neg_data=np.load(path_neg+str(neg_id)+'.npy')#32*11*224*224
        idx2 = random.sample(range(neg_data.shape[0]), batch_num_neg)
        sample_neg_data = np.take(neg_data, idx2, axis=0)
        batch_data.extend(sample_neg_data)
    else:
        ## sample num >= img.shape[0] (64)
        for _ in range(batch_num_pos):
            pos_id=random.randint(1, dc.train_pos_num)
            pos_data=np.load(path_pos+str(pos_id)+'.npy')#32*11*224*224
            batch_data.extend(pos_data)
        for _ in range(batch_num_neg):
            neg_id=random.randint(1, dc.train_neg_num)
            neg_data=np.load(path_neg+str(neg_id)+'.npy')#32*11*224*224
            batch_data.extend(neg_data)

    batch_data=np.asarray(batch_data)#256*11*15*15

    batch_x=batch_data[:,:9]#256*9*15*15
    batch_y=batch_data[:,-1]#256*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#256*1
    batch_y[batch_y > 0]=1

    batch_y = batch_y.reshape(-1, 1)

    return batch_x,batch_y

##采样验证集或测试集
def sample_val_test(img_id, type):
    data = np.load(os.path.join('/workplace/dataset/MODIS_new/', type+'_80/'+str(img_id)+'.npy'))#11*2030*1354

    batch_x=data[:,:9]#12150*9*15*15
    batch_y=data[:,-1]#12150*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#12150*1
    batch_y[batch_y > 0]=1

    batch_y = batch_y.reshape(-1, 1)

    return batch_x,batch_y