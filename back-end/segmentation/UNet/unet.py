""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # class DoubleConv(nn.Module):
# #     """(convolution => [BN] => ReLU) * 2"""

# #     def __init__(self, in_channels, out_channels, mid_channels=None):
# #         super().__init__()
# #         if not mid_channels:
# #             mid_channels = out_channels
# #         self.double_conv = nn.Sequential(
# #             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(mid_channels),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True)
# #         )

# #     def forward(self, x):
# #         return self.double_conv(x)


# # class Down(nn.Module):
# #     """Downscaling with maxpool then double conv"""

# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.maxpool_conv = nn.Sequential(
# #             nn.MaxPool2d(2),
# #             # nn.AdaptiveMaxPool2d(1),
# #             DoubleConv(in_channels, out_channels)
# #         )

# #     def forward(self, x):
# #         return self.maxpool_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super().__init__()
#         self.down = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
#             # nn.BatchNorm2d(out_channels),
#             nn.GroupNorm(out_channels // 16, out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.down(x)


# # class Up(nn.Module):
# #     """Upscaling then double conv"""

# #     def __init__(self, in_channels, out_channels, bilinear=False):
# #         super().__init__()

# #         # if bilinear, use the normal convolutions to reduce the number of channels
# #         if bilinear:
# #             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# #             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
# #         else:
# #             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
# #             self.conv = DoubleConv(in_channels, out_channels)

# #     def forward(self, x1, x2):
# #         x1 = self.up(x1)
# #         # input is CHW
# #         diffY = x2.size()[2] - x1.size()[2]
# #         diffX = x2.size()[3] - x1.size()[3]

# #         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
# #                         diffY // 2, diffY - diffY // 2])
# #         # if you have padding issues, see
# #         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
# #         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
# #         x = torch.cat([x2, x1], dim=1)
# #         return self.conv(x)
# class Up(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
#         super().__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
#             nn.GroupNorm(out_channels//16, out_channels),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x1, x2):
#         # x1 = self.up(x1)
#         # x = torch.cat([x2, x1], dim=1)
#         # return self.conv(x)
#         x = torch.cat((x2, x1), dim=1)
#         return self.up(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_channels=9, n_classes=1):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         filters = [32, 64, 128, 256]
#         self.inc = Down(n_channels, filters[0],kernel_size=1, padding=0)
#         self.down1 = Down(filters[0], filters[1])
#         self.down2 = Down(filters[1], filters[2])
#         self.down3 = Down(filters[2], filters[3])

#         self.down_up1 = Down(filters[3], filters[3])
#         self.down_up2 = Down(filters[3]+filters[3], filters[3])

#         self.up1 = Up(filters[3]+filters[2], filters[2])
#         self.up2 = Up(filters[2]+filters[1], filters[1])
#         self.up3 = Up(filters[1]+filters[0], filters[0])

#         self.outc = (OutConv(filters[0], n_classes))

#     def forward(self, x):
#         x1 = self.inc(x) # [256, 32, 15, 15]
#         x2 = self.down1(x1) # [256, 64, 15, 15])
#         x3 = self.down2(x2) # [256, 128, 15, 15]
#         x4 = self.down3(x3) # [256, 256, 15, 15]
        
#         u = self.down_up1(x4) # [256, 256, 15, 15]
#         x = torch.cat((x4, u), dim=1) #[256, 512, 15, 15]
#         x = self.down_up2(x)

#         x = self.up1(x, x3)
#         x = self.up2(x, x2)
#         x = self.up3(x, x1)


#         logits = self.outc(x)
#         return logits

import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.data
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,ks=3,s=1,pad=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=s, padding=pad, bias=True),
            nn.GroupNorm(out_ch//16, out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,ks=3,s=1,pad=1,op=0):
        super(up_conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ks, stride=s, output_padding=op,padding=pad, bias=True),
            nn.GroupNorm(out_ch//16, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=9):
        super(UNet, self).__init__()
        self.conv1=conv_block(in_ch,32,ks=1,pad=0)
        self.conv2 = conv_block(32, 64, ks=3, pad=1)
        self.conv3 = conv_block(64, 128, ks=3, pad=1)
        self.conv4 = conv_block(128, 256, ks=3, pad=1)
        self.conv_u = conv_block(256, 256, ks=3, s=1, pad=1)

        self.upconv4 = conv_block(256+256, 256, ks=3, s=1, pad=1)
        self.upconv3 = up_conv_block(256+128, 128, ks=3, s=1, pad=1)
        self.upconv2 = up_conv_block(128+64, 64, ks=3, s=1, pad=1)
        self.upconv1 = up_conv_block(64+32, 32, ks=3, s=1,pad=1)

        self.conv = nn.Sequential(
            nn.Conv2d(32,1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            )

    def forward(self, x):
        e1=self.conv1(x) # [256, 32, 15, 15]
        e2=self.conv2(e1) # [256, 64, 15, 15]
        e3 = self.conv3(e2) # [256, 128, 15, 15]
        e4=self.conv4(e3) # [256, 256, 15, 15]
        u = self.conv_u(e4) # [256, 256, 15, 15]
        d4=torch.cat((e4,u),dim=1) # [256, 512, 15, 15]
        d4=self.upconv4(d4) # [256, 256, 15, 15]
        d3=torch.cat((e3,u),dim=1) # [256, 384, 15, 15]
        d3=self.upconv3(d3) # [256, 128, 15, 15]
        d2=torch.cat((e2,d3),dim=1) # [256, 192, 15, 15]
        d2=self.upconv2(d2) # [256, 64, 15, 15]
        d1=torch.cat((e1,d2),dim=1) # [256, 96, 15, 15]
        d1=self.upconv1(d1) # [256, 32, 15, 15]

        out=self.conv(d1) # [256, 1, 15, 15]
        return out

if __name__ == "__main__":
    model = UNet()
    model.cuda()
    model.eval()
    image = torch.randn(256, 9, 15, 15).cuda()

    # print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)