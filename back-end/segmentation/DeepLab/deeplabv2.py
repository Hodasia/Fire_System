#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import ConvBnReLU, Stem, ResLayer

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", Stem(ch[0]))
        self.add_module("layer2", ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        # self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer3", ResLayer(n_blocks[1], ch[2], ch[3], 1, 1))
        self.add_module("layer4", ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
        self.add_module("sigmoid", nn.Sigmoid())

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, ConvBnReLU.BATCH_NORM):
                m.eval()

# if __name__ == "__main__":
#     model = DeepLabV2(
#         n_classes=1, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
#     )
#     model.cuda()
#     model.eval()
#     image = torch.randn(256, 9, 15, 15).cuda()

#     # print(model)
#     print("input:", image.shape)
#     print("output:", model(image).shape)