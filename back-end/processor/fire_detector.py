import os
import json
import numpy as np
import math
import torch
from tqdm import trange
import models.simplevit as simplevit
import models.unet as unet
import utils.funcs as funcs
import utils.norm as norm
import utils.metrics as metrics

data_path = 'G:/1/data/'
origin_path = 'G:/1/origin/'
results_path = 'G:/1/results/'
score_path = 'score.json'

h,w=2025,1350
ks=15
skip=ks
cl=funcs.chunk_list(h,w,ks,skip)
output_channel = 9
thrd=0.5
batch_num=10
c21_thrd, c22_thrd=1.0, 1.1

dict_list=[]

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

for img_file in os.listdir(data_path):
    # 输入原图
    img_path = os.path.join(data_path, img_file)
    data = np.load(img_path)
    img = data['arr_0'][:, 0:h, 0:w]
    origin_img = img[output_channel]

    img_id=os.path.splitext(img_file)[0]
    output_file = img_id + '.png'

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

    tp, fp, tn, fn=0, 0, 0, 0

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
    print(f1)
    if(f1<0.86):
        continue
    
    dict_list.append(img_info)
    file_name = 'score.json'
    # 将字典写入JSON文件
    with open(file_name, 'w') as json_file:
        json.dump(dict_list, json_file, indent=4)

    output_origin = os.path.join(origin_path, output_file)
    funcs.print_img(origin_img, output_origin, 'origin')
    output_results = os.path.join(results_path, output_file)
    funcs.print_img(restored_img, output_results, 'results')


file_name = 'score.json'
# 将字典写入JSON文件
with open(file_name, 'w') as json_file:
    json.dump(dict_list, json_file, indent=4)