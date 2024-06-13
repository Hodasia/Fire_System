import torch
import torch.nn as nn
import os, sys
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
from utils.init_weights import *

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, n=2, kernel_size=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n=n
        
        for i in range(1, n+1):
            conv = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.GroupNorm(out_size // 16, out_size),
                nn.ReLU(inplace=True))
            setattr(self, 'conv%d'%i, conv)
            in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, n_concat=2, kernel_size=3, stride=1, padding=1, output_padding=0):
        super(unetUp, self).__init__()
        # self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.GroupNorm(out_size//16, out_size),
            nn.ReLU(inplace=True)
        )

        # if self.is_deconv:
        #     self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1, output_padding=0)
        #     # self.up = nn.Sequential(
        #     #     nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=1, padding=1, output_padding=0),
        #     #     nn.BatchNorm2d(out_size),
        #     #     nn.ReLU(inplace=True))
            
        # else:
        #     self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=3, stride=2, padding=0, output_padding=0)
        #     # self.up = nn.Sequential(
        #     #      nn.UpsamplingBilinear2d(scale_factor=2.2),
        #     #      nn.Conv2d(in_size, out_size, 1))
            
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = high_feature
        # outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        # return self.conv(outputs0)
        return self.up(outputs0)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=9, n_classes=1, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.is_ds = is_ds

        filters = [32, 64, 128, 256, 512]
        # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        # self.pad = nn.ConstantPad2d((1, 0, 1, 0), 0.0)
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.adaptivepool = nn.AdaptiveMaxPool2d(14)
        self.conv00 = unetConv2(self.in_channels, filters[0], kernel_size=1, padding=0)
        self.conv10 = unetConv2(filters[0], filters[1])
        self.conv20 = unetConv2(filters[1], filters[2])
        self.conv30 = unetConv2(filters[2], filters[3])
        self.conv40 = unetConv2(filters[3], filters[4])

        # upsampling
        self.up_concat01 = unetUp(filters[1]+filters[0], filters[0])
        self.up_concat11 = unetUp(filters[2]+filters[1], filters[1])
        self.up_concat21 = unetUp(filters[3]+filters[2], filters[2])
        self.up_concat31 = unetUp(filters[4]+filters[3], filters[3])

        self.up_concat02 = unetUp(filters[1]+filters[0]*2, filters[0], 3)
        self.up_concat12 = unetUp(filters[2]+filters[1]*2, filters[1], 3)
        self.up_concat22 = unetUp(filters[3]+filters[2]*2, filters[2], 3)

        self.up_concat03 = unetUp(filters[1]+filters[0]*3, filters[0], 4)
        self.up_concat13 = unetUp(filters[2]+filters[1]*3, filters[1], 4)
        
        self.up_concat04 = unetUp(filters[1]+filters[0]*4, filters[0], 5)
        
        # # final conv (without any concat)
        # self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # final conv (without any concat)
        # self.final_1 = OutConv(filters[0], n_classes)
        # self.final_2 = OutConv(filters[0], n_classes)
        # self.final_3 = OutConv(filters[0], n_classes)
        self.final_4 = OutConv(filters[0], n_classes)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        # inputs = self.pad(inputs)
        X_00 = self.conv00(inputs)         # 32*15*15  
        # maxpool0 = self.maxpool(X_00)    # 32*7*7
        # X_10= self.conv10(maxpool0)      
        X_10= self.conv10(X_00)            # 64*15*15 
        # maxpool1 = self.maxpool(X_10)      
        # X_20 = self.conv20(maxpool1)     
        X_20 = self.conv20(X_10)           # 128*7*7
        # maxpool2 = self.maxpool(X_20)        
        # X_30 = self.conv30(maxpool2)        
        X_30 = self.conv30(X_20)           # 256*15*15
        # maxpool3 = self.maxpool(X_30)       
        # X_40 = self.conv40(maxpool3)        
        X_40 = self.conv40(X_30)           # 512*15*15
        
        # column : 1
        X_01 = self.up_concat01(X_10,X_00) # 32*15*15
        X_11 = self.up_concat11(X_20,X_10) # 64*15*15
        X_21 = self.up_concat21(X_30,X_20) # 128*15*15
        X_31 = self.up_concat31(X_40,X_30) # 256*15*15

        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01) # 32*15*15
        X_12 = self.up_concat12(X_21,X_10,X_11) # 64*15*15
        X_22 = self.up_concat22(X_31,X_20,X_21) # 128*15*15

        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02) # 32*15*15
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12) # 64*15*15
        
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03) # 32*15*15

        # final layer
        # final_1 = self.final_1(X_01)
        # final_2 = self.final_2(X_02)
        # final_3 = self.final_3(X_03)
        # final_4 = self.final_4(X_04)

        # final = (final_1+final_2+final_3+final_4)/4
        # final = (final_1+final_2+final_3)/3

        # if self.is_ds:
        #     return final
        # else:
        #     # return final_3
            # return final_4

            
        final_4 = self.final_4(X_04)
        return final_4
        # final_3 = self.final_3(X_03)
        # return final_3

if __name__ == "__main__":
    model = UNet_Nested()
    model.cuda()
    model.eval()
    image = torch.randn(256, 9, 15, 15).cuda()

    # print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)

      