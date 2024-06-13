import numpy as np
## 统计tp, fp, fn
def cal_hit(pred,true):
    #0/1:b,1

    # print(pred.flatten().tolist().count(0),pred.flatten().tolist().count(1))
    # print(true.flatten().tolist().count(0), true.flatten().tolist().count(1))

    ##tp:p+t=2
    ##tn:p+t=0
    tcnt=pred+true
    tp=tcnt.flatten().tolist().count(2)
    tn=tcnt.flatten().tolist().count(0)

    ##fp:p-t=1
    fp=pred-true
    fp = fp.flatten().tolist().count(1)

    ##fn:p-t=-1
    fn = pred-true
    fn = fn.flatten().tolist().count(-1)

    return tp,fp,tn,fn

# 计算IoU
# def cal_iou(pred, true):
#     inter = np.logical_and(pred, true).sum()
#     union = np.logical_or(pred, true).sum()

#     iou = inter / union if union > 0 else 0.0
#     return iou

def cal_score(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    acc = 0 if total == 0 else (tp + tn) / total

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    iou = p * r / (p + r - p * r) if p + r - p * r > 0 else 0

    return f1, p, r, iou, acc
