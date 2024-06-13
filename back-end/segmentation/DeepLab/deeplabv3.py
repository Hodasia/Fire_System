#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ConvBnReLU, Stem, ResLayer

class ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        # h = self.pool(x)
        h = self.conv(x)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )
        self.stages.add_module("imagepool", ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4], output_stride=8):
        super(DeepLabV3, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 1, 1, 1]
            # s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 1, 1, 1]
            # s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", Stem(ch[0]))
        self.add_module("layer2", ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0]))
        self.add_module("layer3", ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1]))
        self.add_module("layer4", ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2]))
        self.add_module(
            "layer5", ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        )
        self.add_module("aspp", ASPP(ch[5], 256, atrous_rates))
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))
        self.add_module("sigmoid", nn.Sigmoid())

# if __name__ == "__main__":
#     model = DeepLabV3(
#         n_classes=1,
#         n_blocks=[3, 4, 23, 3],
#         atrous_rates=[6, 12, 18],
#         multi_grids=[1, 2, 4],
#         output_stride=8,
#     )
#     model.cuda()
#     model.eval()
#     image = torch.randn(256, 9, 15, 15).cuda()

#     # print(model)
#     print("input:", image.shape)
#     print("output:", model(image).shape)