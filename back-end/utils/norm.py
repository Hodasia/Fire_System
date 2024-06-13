import numpy as np
import utils.data_config as data_config
dc_classify = data_config.DataConfig_Classify
dc_seg = data_config.DataConfig_Seg

## 归一化
def nor2(x, flag):
    b, _, h, w = x.shape
    num_channels = x.shape[1]

    x_new = np.zeros((b, num_channels, h, w))

    if flag == 'classify':
        dc = dc_classify
    elif flag == 'seg':
        dc = dc_seg
    else:
        print('error in norm')

    for i in range(num_channels):
        min_val = getattr(dc, f"min{i}")
        max_val = getattr(dc, f"max{i}")
        x_new[:, i] = (x[:, i] - min_val) / (max_val - min_val)

    return x_new