import os
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import torch
import torch.nn as nn
import sys,os
from datetime import datetime

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)

import utils.data_config as data_config
import utils.logger as logger
import utils.metrics as metrics
import utils.norm as norm
import utils.sample_data as sample_data
import setr

dc=data_config.DataConfig

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


## 评估模型
def evaluate(args, model, data_type):
    tp, fp, tn, fn=0, 0, 0, 0

    index=1
    total_num=args.test_total_num
    batch_num=args.test_batch
    thrd=args.evaluate_threshold
    
    model.eval()
    with torch.no_grad():
        while(index<=total_num):
            idx_list = [index + i for i in range(0, batch_num)]
            test_x, test_y = sample_data.sample_val_test(idx_list, data_type)
            input = norm.nor2(test_x)
            input = torch.Tensor(input).cuda()
            
            pred = model(input)  # 8*9*101*101*1
            pred=pred.detach().cpu().numpy()
            pred[pred<=thrd]=0
            pred[pred>thrd]=1

            tpp,fpp,tnn,fnn= metrics.cal_hit(pred,test_y)
            tp+=tpp
            fp+=fpp
            tn+=tnn
            fn+=fnn

            index += batch_num

    return tp,fp,tn,fn

def train():
    set_seeds(args)
    best_val_perf = float('-inf')
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))

    model = setr.SETR_Naive()
    
    model.cuda()

    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)

    log.log('***** train *****')
    log.log('# epochs: {:d}'.format(args.epochs))
    log.log('# train batch: {:d}'.format(args.train_batch))
    log.log('# sample size : {:d}'.format(args.sample))
    log.log('# learning rate: {:g}'.format(args.lr))
    log.log('# parameters: {:d}'.format(count_parameters(model)))
    log.log('***** valid *****')
    log.log('# valid total num: {:d}'.format(args.test_total_num))
    log.log('# valid batch: {:d}'.format(args.test_batch))
    log.log('# threshold: {:g}'.format(args.evaluate_threshold))
    log.log('***** test *****')
    log.log('# test epoch total num: {:d}'.format(args.test_total_num))
    log.log('# test batch: {:d}'.format(args.test_batch))
    log.log('# threshold: {:g}'.format(args.evaluate_threshold))

    epoch=1
    tr_loss, logging_loss = 0.0, 0.0
    step_num = 0
    current_best = 0.0
    curr_best_epoch = 0
    early_stop_cnt = 0
    early_stop_flag = False

    for i in range(1, args.sample * args.epochs + 1):
        if(i%args.sample==1):
            epoch_start_time = datetime.now()
            epoch_train_start_time = datetime.now()
        
        # 训练
        model.train()
        train_x,train_y= sample_data.sample_train(args.batch_num, args.cut_batch)

        ims = norm.nor2(train_x)  # ims/255
        ims = torch.Tensor(ims).cuda()
        
        pred = model(ims)
        loss = criterion(pred, torch.Tensor(train_y).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_num += 1
        tr_loss += loss.detach()

        # sample为一个epoch的采样次数，即batch的数量
        # 验证
        if(i%args.sample==0):

            log.log('#####Epoch:{:3d}######'.format(epoch))
            log.log('training time is {:s}, train loss {:8.4f}'.format(strtime(epoch_train_start_time), tr_loss / step_num))
            
            scheduler.step()
            tp, fp, tn, fn = evaluate(args, model, 'valid')

            log.log('Eval Valid tp:{:3d}  fp:{:3d}  tn:{:3d}  fn:{:3d}'.format(tp, fp, tn, fn))

            f1, p, r, iou, acc = metrics.cal_score(tp, fp, tn, fn)
            log.log('f1:{:8.4f}  precision:{:8.4f}  recall:{:8.4f}  iou:{:8.4f}  acc:{:8.4f}\n'
            ' validation time {} '.format(f1, p, r, iou, acc, strtime(epoch_start_time)))

            if f1 > best_val_perf:
                early_stop_cnt = 0
                current_best = f1
                curr_best_epoch = epoch
                log.log('------- new best val perf: {:g} --> {:g}'.format(best_val_perf, current_best))

                best_val_perf = current_best
                torch.save({'opt': args,
                            'sd': model.state_dict(),
                            'perf': best_val_perf, 'epoch': epoch,
                            'opt_sd': optimizer.state_dict(),
                            'tr_loss': tr_loss, 'step_num': step_num,
                            'logging_loss': logging_loss},
                        args.model)
            else:
                early_stop_cnt += 1
                log.log(f'EarlyStopping counter: {early_stop_cnt} out of {args.patience}')
                if early_stop_cnt >= args.patience:
                    early_stop_flag = True

            if early_stop_flag:
                log.log("EarlyStopping: Stop training")
                break
            epoch += 1
    
    log.log('best f1 {:g} in epoch {:g}'.format(current_best, curr_best_epoch))

def test():
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))

    model = setr.SETR_Naive()

    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict['sd'])
    model.cuda()

    tp, fp, tn, fn = evaluate(args, model, 'test')
    log.log('Eval Test tp:{:3d}  fp:{:3d}  tn{:3d}  fn:{:3d}'.format(tp, fp, tn, fn))
    
    f1, p, r, iou, acc = metrics.cal_score(tp, fp, tn, fn)
    log.log('f1:{:8.4f}  precision:{:8.4f}  recall:{:8.4f}  iou:{:8.4f}, acc:{:8.4f}'.format(f1, p, r, iou, acc))

if __name__ == '__main__':
    print('********* START *********')
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        default="/workplace/project/Fire_MODIS_v3/segmentation/SETR/models_save/setr_1.pth")

    parser.add_argument("--batch_num", default=4)
    parser.add_argument("--cut_batch", default=False)

    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--train_batch", default=256)
    parser.add_argument("--sample", default=100)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--step_size", default=20)
    parser.add_argument("--scheduler_gamma", default=0.1)

    parser.add_argument("--test_total_num", default=1374)
    parser.add_argument("--test_batch", default=6)
    parser.add_argument("--evaluate_threshold", default=0.5)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--patience", default=50)

    args = parser.parse_args()

    train()
    test()