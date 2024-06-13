# utils:
1. 评价指标
2. normalization
3. 数据采样

# dataProcess
1. 数据下载
2. 训练集测试集怎么处理和划分

# ResNet, DenseNet, ViT
1. main.py:train函数、验证函数、测试函数
2. model.py:模型
3. models_save:保存模型

# dataset:
1. train_80: 
    + (64, 11, 15, 15) batch_size=64
    + pos: 4843
    + neg: 8335
    + pos:neg = 1:1.7
2. valid_80: 
    + (12150, 11, 15, 15) batch_size为一张全图经过c21=1.0,c22=1.1阈值筛选过后剩下的子图数量
    + 1074 张
3. test_80: 
    + (12150, 11, 15, 15) batch_size为一张全图经过c21=1.0,c22=1.1阈值筛选过后剩下的子图数量
    + 1074 张
    