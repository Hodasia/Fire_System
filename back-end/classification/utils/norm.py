import numpy as np
import utils.data_config as data_config
dc = data_config.DataConfig

## 归一化
def nor2(x):
    b, _, h, w = x.shape
    num_channels = x.shape[1]

    x_new = np.zeros((b, num_channels, h, w))

    for i in range(num_channels):
        min_val = getattr(dc, f"min{i}")
        max_val = getattr(dc, f"max{i}")
        x_new[:, i] = (x[:, i] - min_val) / (max_val - min_val)

    return x_new