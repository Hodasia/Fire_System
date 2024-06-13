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