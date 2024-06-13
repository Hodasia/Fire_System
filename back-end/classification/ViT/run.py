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
import math

path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)

import utils.data_config as data_config
import utils.logger as logger
import utils.metrics as metrics
import utils.norm as norm
import utils.sample_data as sample_data
import vit
import simplevit

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
    num=args.test_epochs
    thrd=args.evaluate_threshold
    batch_num=args.test_batch
    
    model.eval()
    with torch.no_grad():
        while(index<=num):
            test_x, test_y = sample_data.sample_val_test(index, data_type)
            test_x = norm.nor2(test_x)
            cnt = math.ceil(test_x.shape[0]/batch_num)
            new_test_x = np.array_split(test_x, cnt, axis=0)
            new_test_y = np.array_split(test_y, cnt, axis=0)

            for i in range(len(new_test_x)):
                input = new_test_x[i]
                input = torch.Tensor(input).cuda()

                pred = model(input)  # 8*9*101*101*1
                pred=pred.detach().cpu().numpy()
                pred[pred<=thrd]=0
                pred[pred>thrd]=1

                tpp,fpp,tnn,fnn= metrics.cal_hit(pred,new_test_y[i])
                tp+=tpp
                fp+=fpp
                tn+=tnn
                fn+=fnn

            index += 1

    return tp,fp,tn,fn

def train():
    set_seeds(args)
    best_val_perf = float('-inf')
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))

    # model = vit.ViT()
    model = simplevit.SimpleViT()
    
    model.cuda()

    # criterion = nn.NLLLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)

    log.log('***** train *****')
    log.log('# epochs: {:d}'.format(args.epochs))
    log.log('# train batch: {:d}'.format(args.train_batch))
    log.log('# sample size : {:d}'.format(args.sample))
    log.log('# learning rate: {:g}'.format(args.lr))
    log.log('# parameters: {:d}'.format(count_parameters(model)))
    log.log('***** valid *****')
    log.log('# valid epoch: {:d}'.format(args.test_epochs))
    log.log('# valid batch: {:d}'.format(args.test_batch))
    log.log('# threshold: {:g}'.format(args.evaluate_threshold))
    log.log('***** test *****')
    log.log('# test epoch: {:d}'.format(args.test_epochs))
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
        train_x,train_y= sample_data.sample_train(args.batch_num_pos, args.batch_num_neg, args.cut_batch)

        ims = norm.nor2(train_x)  # ims/255
        ims = torch.Tensor(ims).cuda()
        
        pred = model(ims)
        # loss = criterion(pred, torch.Tensor(train_y).long().cuda())
        loss = criterion(pred, torch.Tensor(train_y).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_num += 1
        tr_loss += loss.detach()

        # sample为一个epoch的采样次数，即batch的数量
        # 验证
        if(i%args.sample==0):

            log.log('training time for epoch {:3d} '
                   'is {:s}'.format(epoch, strtime(epoch_train_start_time)))
            
            scheduler.step()
            tp, fp, tn, fn = evaluate(args, model, 'valid')

            log.log('Eval: Valid')
            log.log('tp:{:3d}  fp:{:3d}  tn:{:3d}  fn:{:3d}'.format(tp, fp, tn, fn))
            acc=(tp+tn)/(tp+fp+tn+fn)
            if(tp+fp>0 and tp+fn>0):
                p=tp/(tp+fp)
                r=tp/(tp+fn)
                if(p+r>0):
                    f1=2*p*r/(p+r)
                    fa=1-p
                    log.log('Done with epoch:{:3d}  train loss {:8.4f}\n '
                   'f1:{:8.4f}  precision:{:8.4f}  recall:{:8.4f}  fa:{:8.4f}  acc:{:8.4f}\n'
                   ' epoch time {} '.format(epoch, tr_loss / step_num, f1, p, r, fa, acc, strtime(epoch_start_time)))

                    if f1 > best_val_perf:
                        early_stop_cnt = 0
                        current_best = f1
                        curr_best_epoch = epoch
                        log.log('------- new best val perf: {:g} --> {:g} '
                                ''.format(best_val_perf, current_best))

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
                        if early_stop_cnt > args.patience:
                            early_stop_flag = True

            if early_stop_flag:
                log.log("EarlyStopping: Stop training")
                break
            epoch += 1
    
    log.log('best f1 {:g} in epoch {:g}'.format(current_best, curr_best_epoch))

def test():
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))

    # model=vit.ViT()
    model=simplevit.SimpleViT()

    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict['sd'])
    model.cuda()

    tp, fp, tn, fn = evaluate(args, model, 'test')

    log.log('Eval: Test')
    log.log('tp:{:3d}  fp:{:3d}  tn{:3d}  fn:{:3d}'.format(tp, fp, tn, fn))
    if (tp + fp > 0 and tp + fn > 0):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if (p + r > 0):
            f1 = 2 * p * r / (p + r)
            fa=1-p
            log.log('f1:{:8.4f}  precision:{:8.4f}  recall:{:8.4f}  fa:{:8.4f}'.format(f1, p, r, fa))

if __name__ == '__main__':
    print('********* START *********')
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",
                        default="/workplace/project/Fire_MODIS_v3/classification/ViT/models_save/simpleViT_2.pth")
    # parser.add_argument("--model",
    #                     default="/workplace/project/Fire_MODIS_v2/ViT/simpleViT_80_2.pth")
    
    parser.add_argument("--batch_num_pos", default=2)
    parser.add_argument("--batch_num_neg", default=2)
    parser.add_argument("--cut_batch", default=False)

    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--train_batch", default=256)
    parser.add_argument("--sample", default=100)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--step_size", default=20)
    parser.add_argument("--scheduler_gamma", default=0.1)

    parser.add_argument("--test_epochs", default=1074)
    parser.add_argument("--test_batch", default=256)
    parser.add_argument("--evaluate_threshold", default=0.5)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--patience", default=50)

    args = parser.parse_args()

    train()
    # test()