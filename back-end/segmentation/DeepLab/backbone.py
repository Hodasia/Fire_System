import torch
import torch.nn as nn
import torch.nn.functional as F

# try:
#     from encoding.nn import SyncBatchNorm

#     _BATCH_NORM = SyncBatchNorm
# except:
#     _BATCH_NORM = nn.BatchNorm2d

_BATCH_NORM = nn.BatchNorm2d
_BOTTLENECK_EXPANSION = 4



class ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class Bottleneck(nn.Module):
    """
    Bottleneck block of MSRA ResNet.
    """

    def __init__(self, in_ch, out_ch, stride, dilation, downsample):
        super(Bottleneck, self).__init__()
        mid_ch = out_ch // _BOTTLENECK_EXPANSION
        self.reduce = ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class ResLayer(nn.Sequential):
    """
    Residual layer with multi grids
    """

    def __init__(self, n_layers, in_ch, out_ch, stride, dilation, multi_grids=None):
        super(ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(multi_grids)

        # Downsampling is only in the first block
        for i in range(n_layers):
            self.add_module(
                "block{}".format(i + 1),
                Bottleneck(
                    in_ch=(in_ch if i == 0 else out_ch),
                    out_ch=out_ch,
                    stride=(stride if i == 0 else 1),
                    dilation=dilation * multi_grids[i],
                    downsample=(True if i == 0 else False),
                ),
            )


class Stem(nn.Sequential):
    """
    The 1st conv layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self, out_ch):
        super(Stem, self).__init__()
        self.add_module("conv1", ConvBnReLU(9, out_ch, 3, 1, 1, 1))
        # self.add_module("conv1", _ConvBnReLU(9, out_ch, 7, 2, 3, 1))
        # self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))
