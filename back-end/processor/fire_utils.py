import cv2
import os
import json
import shutil
import numpy as np
from matplotlib import pyplot as plt
import math
import torch
from tqdm import trange
import models.simplevit as simplevit
import models.unet as unet
import utils.funcs as funcs
import utils.norm as norm
import utils.metrics as metrics

def read_img_info(img_id, file_name):
    # 读取JSON文件中的数据
    
    with open(file_name, 'r') as json_file:
        data = json.load(json_file)
    
    # 根据img_id查找对应的字典数据
    img_info = None
    for item in data:
        if item['id'] == img_id:
            img_info = item
            break
    
    return img_info

# 读取本地结果
def save_results(img_name):
    img_id = img_name.rsplit('.', 1)[0]
    ext = 'png'

    origin_src = os.path.join('./processor/final/origin', img_id+'.'+ext)
    shutil.copy(origin_src, './tmp/ct')

    results_src = os.path.join('./processor/final/results', img_id+'.'+ext)
    shutil.copy(results_src, './tmp/draw')

    file_name = './processor/final/score.json'
    img_info = read_img_info(img_id, file_name)

    return img_info

# 切分子图索引
def chunk_list(h,w,ks,skip):
    hi,wi=0,0
    chunk_list=[]
    while(hi+ks<=h):
        if(wi+ks<=w):
            tmp=[hi,hi+ks,wi,wi+ks]
            chunk_list.append(tmp)
            wi+=skip
        else:
            wi=0
            hi+=skip
    return chunk_list

# 打印结果
def print_img(img, path, flag):
    plt.figure(figsize=(13.25,20.5),dpi=100)
    if flag == 'origin':
        plt.imshow(img, cmap=plt.cm.gist_earth)
    elif flag == 'results':
        plt.imshow(img, cmap=plt.cm.gist_stern)
    # plt.colorbar()
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)
    # plt.show()

def show_sub(img_id, ks, cnt):
    # ks=15*5
    skip=ks
    h,w=2025,1350
    cl=chunk_list(h,w,ks,skip)

    ext = 'npz'
    data_path = os.path.join('./processor/final/data/', img_id +'.'+ ext)
    data = np.load(data_path)
    img = data['arr_0'][:,0:h, 0:w]

    img = img[10]
    img[img<80]=0
    img[img>=80]=1

    sub_img = []
    for c in cl:
        d=img[c[0]:c[1],c[2]:c[3]]
        if np.all(d == 0):
            continue
        sub_img.append(d)
    sub_img = np.asarray(sub_img)

    num_ones = np.sum(sub_img, axis=(1, 2))
    max_index = np.argmax(num_ones)

    img_path = os.path.join('./tmp/sub/', img_id + '_' + str(cnt) + '.png')
    # plt.figure(figsize=(0.75,0.75),dpi=100)
    plt.imshow(sub_img[max_index], cmap=plt.cm.flag_r)
    plt.axis('off')
    plt.savefig(img_path,bbox_inches='tight', pad_inches = -0.1)
    
    return cl[max_index][0], cl[max_index][1], cl[max_index][2], cl[max_index][3]

def fire_detect(img_path, img_id):
    h,w=2025,1350
    ks=15
    skip=ks
    cl=chunk_list(h,w,ks,skip)
    output_channel = 9
    thrd=0.5
    batch_num=10
    c21_thrd, c22_thrd=1.0, 1.1

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    # 分类模型加载
    model_simplevit = simplevit.SimpleViT()
    checkpoint_simplevit = torch.load('./weights/simpleViT_1.pth')
    model_simplevit.load_state_dict(checkpoint_simplevit['sd'])
    model_simplevit.cuda()
    model_simplevit.eval()

    # 分割模型加载
    model_unet = unet.UNet()
    checkpoint_unet = torch.load('./weights/unet_5.pth')
    model_unet.load_state_dict(checkpoint_unet['sd'])
    model_unet.cuda()
    model_unet.eval()

    data = np.load(img_path)
    img = data['arr_0'][:, 0:h, 0:w]
    origin_img = img[output_channel]

    # 切成子图
    sub_img = []
    for c in cl:
        d=img[:, c[0]:c[1],c[2]:c[3]]
        sub_img.append(d)
    sub_img = np.asarray(sub_img)

    # 分类
    x_cls = sub_img[:, :9]
    output_cls = []

    # 阈值初筛
    for i in range(sub_img.shape[0]):
        p_c21, p_c22 = np.max(sub_img[i, 4]), np.max(sub_img[i, 5])
        if not (p_c21 >= c21_thrd and p_c22 >= c22_thrd):
            output_cls.extend([0])
        else:
            output_cls.extend([1])

    output_cls = np.asarray(output_cls)
    print(output_cls.shape)

    # 送入分类模型进行进一步筛选
    indices_cls = np.where(output_cls == 1)[0]
    print(indices_cls.shape)
    new_tmp = norm.nor2(x_cls[indices_cls], 'classify')
    cnt = math.ceil(new_tmp.shape[0]/batch_num)
    new_x_cls = np.array_split(new_tmp, cnt, axis=0)
    output = []

    with torch.no_grad():
        for i in trange(len(new_x_cls)):
            input_cls = new_x_cls[i]
            input_cls = torch.Tensor(input_cls).cuda()

            pred_cls = model_simplevit(input_cls)
            pred_cls = pred_cls.detach().cpu().numpy()
            pred_cls[pred_cls<=thrd]=0
            pred_cls[pred_cls>thrd]=1
            output.extend(pred_cls)
    output = np.asarray(output)
    print(output.shape)

    output_cls = output_cls.reshape(-1,1)
    output_cls[indices_cls] = output

    # 分割
    indices_seg = np.where(output_cls == 1)[0] # 只对火点子图进行分割
    x_seg = sub_img[indices_seg][:, :9]
    y_seg = sub_img[indices_seg][:, -1]
    y_seg[y_seg<80]=0
    y_seg[y_seg>=80]=1
    y_seg = np.expand_dims(y_seg, axis=1)

    output_seg = []

    with torch.no_grad():
        input_seg = norm.nor2(x_seg, 'seg')
        input_seg = torch.Tensor(input_seg).cuda()
                
        pred_seg = model_unet(input_seg)
        pred_seg=pred_seg.detach().cpu().numpy()
        pred_seg[pred_seg<=thrd]=0
        pred_seg[pred_seg>thrd]=1
        output_seg.extend(pred_seg)
    output_seg = np.asarray(output_seg)

    cnt_fire = 0
    tp, fp, tn, fn=0, 0, 0, 0

    # 还原图像
    # 创建一个全零数组，形状与原始图像相同
    restored_img = np.zeros((h,w))

    # 将 sub_img 中的子图像放回原始图像对应的位置
    for i, c in enumerate(cl):
        if i in indices_seg:
            restored_sub = output_seg[cnt_fire].reshape((15, 15))
            cnt_fire += 1
        else:
            restored_sub = np.zeros((15, 15))
        restored_img[c[0]:c[1], c[2]:c[3]] = restored_sub

    con = img[-1, :]
    con[con < 80] = 0
    con[con >= 80] = 1
    print(con.shape)
    print(restored_img.shape)
    tp,fp,tn,fn= metrics.cal_hit(restored_img,con)
    f1, p, r, iou, acc = metrics.cal_score(tp, fp, tn, fn)
    print(f1, p, r, iou, acc)

    img_info = {'id': img_id, 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'f1': round(f1, 4), 'p': round(p, 4), 'r': round(r, 4), 'iou': round(iou, 4), 'acc': round(acc, 4)}
    print(tn,fn)
    print(fp,tp)

    return origin_img, restored_img, img_info
