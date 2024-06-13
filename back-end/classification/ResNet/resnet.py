import torch
import torch.nn as nn

# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# ResNet-18/34 残差结构 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1   # 残差结构中主分支所采用的卷积核的个数是否发生变化。对于浅层网络，每个残差结构的第一层和第二层卷积核个数一样，故是1

    # 定义初始函数
    # in_channel输入特征矩阵深度，out_channel输出特征矩阵深度（即主分支卷积核个数）
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_cbam=False):   # downsample对应虚线残差结构捷径中的1×1卷积
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 使用bn层时不使用bias
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)  # 实/虚线残差结构主分支中第二层stride都为1
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample   # 默认是None
        self.use_cbam = use_cbam

        if self.use_cbam:
            self.ca = ChannelAttention(out_channel*self.expansion)
            self.sa = SpatialAttention()

# 定义正向传播过程
    def forward(self, x):
        identity = x   # 捷径分支的输出值
        if self.downsample is not None:   # 对应虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)   # 这里不经过relu激活函数

        if self.use_cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out += identity
        out = self.relu(out)

        return out

# ResNet-50/101/152 残差结构 Bottleneck
class Bottleneck(nn.Module):
    expansion = 4    # 第三层的卷积核个数是第一层、第二层的四倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,   # out_channels是第一、二层的卷积核个数
                               kernel_size=1, stride=1, bias=False)  # squeeze channels  高和宽不变
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)   # 实线stride为1，虚线stride为2
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,    # 卷积核个数为4倍
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.use_cbam = use_cbam

        if self.use_cbam:
            self.ca = ChannelAttention(out_channel*self.expansion)
            self.sa = SpatialAttention()

# 正向传播过程
    def forward(self, x):
        identity = x
        if self.downsample is not None:   # 对应虚线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.ca(out) * out
            out = self.sa(out) * out

        out += identity
        out = self.relu(out)

        return out


# ResNet整个网络的框架部分
class ResNet(nn.Module):

    def __init__(self,
                 block,   # 残差结构，Basicblock or Bottleneck
                 blocks_num,   # 列表参数，所使用残差结构的数目，如对ResNet-34来说即是[3,4,6,3]
                 num_classes=1,   # 训练集的分类个数
                 include_top=True,# 为了能在ResNet网络基础上搭建更加复杂的网络，默认为True
                 use_cbam=False):   
        super(ResNet, self).__init__()
        self.include_top = include_top   # 传入类变量

        self.in_channel = 64   # 通过max pooling之后所得到的特征矩阵的深度

        self.conv1 = nn.Conv2d(9, self.in_channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # self.ca1 = ca_sa.ChannelAttention(in_planes=self.in_channel)
        # self.sa1 = ca_sa.SpatialAttention()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 高和宽缩减为原来的一半

        self.layer1 = self._make_layer(block, 64, blocks_num[0], use_cbam=use_cbam)   # 对应conv2_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, use_cbam=use_cbam)   # 对应conv3_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, use_cbam=use_cbam)   # 对应conv4_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=1, use_cbam=use_cbam)   # 对应conv5_x

        # self.ca2 = ca_sa.ChannelAttention(in_planes=self.in_channel)
        # self.sa2 = ca_sa.SpatialAttention()

        if self.include_top:   # 默认为True
            # 无论输入特征矩阵的高和宽是多少，通过自适应平均池化下采样层，所得到的高和宽都是1
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)   # num_classes为分类类别数
            self.sigmoid=nn.Sigmoid()
            # self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.modules():   # 卷积层的初始化操作
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1, use_cbam=False):   # stride默认为1
        # block即BasicBlock/Bottleneck
        # channel即残差结构中第一层卷积层所使用的卷积核的个数
        # block_num即该层一共包含了多少层残差结构
        downsample = None

        # 左：输出的高和宽相较于输入会缩小；右：输入channel数与输出channel数不相等
        # 两者都会使x和identity无法相加
        if stride != 1 or self.in_channel != channel * block.expansion:  # ResNet-18/34会直接跳过该if语句（对于layer1来说）
            # 对于ResNet-50/101/152：
            # conv2_x第一层也是虚线残差结构，但只调整特征矩阵深度，高宽不需调整
            # conv3/4/5_x第一层需要调整特征矩阵深度，且把高和宽缩减为原来的一半
            downsample = nn.Sequential(       # 下采样
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))   # 将特征矩阵的深度翻4倍，高和宽不变（对于layer1来说）

        layers = []
        layers.append(block(self.in_channel,  # 输入特征矩阵深度，64
                            channel,  # 残差结构所对应主分支上的第一个卷积层的卷积核个数
                            downsample=downsample,
                            stride=stride, use_cbam=use_cbam))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):   # 从第二层开始都是实线残差结构
            layers.append(block(self.in_channel,  # 对于浅层一直是64，对于深层已经是64*4=256了
                                channel, use_cbam=use_cbam))  # 残差结构主分支上的第一层卷积的卷积核个数
        
        # 通过非关键字参数的形式传入nn.Sequential
        return nn.Sequential(*layers)   # *加list或tuple，可以将其转换成非关键字参数，将刚刚所定义的一切层结构组合在一起并返回

# 正向传播过程
    def forward(self, x):
        x = self.conv1(x)   # 3×3卷积层
        x = self.bn1(x)
        x = self.relu(x)

        # x = self.maxpool(x)    # 3×3 max pool

        x = self.layer1(x)   # conv2_x所对应的一系列残差结构
        x = self.layer2(x)   # conv3_x所对应的一系列残差结构
        x = self.layer3(x)   # conv4_x所对应的一系列残差结构
        x = self.layer4(x)   # conv5_x所对应的一系列残差结构

        if self.include_top:
            x = self.avgpool(x)    # 平均池化下采样
            x = torch.flatten(x, 1)    
            x = self.fc(x)
            x = self.sigmoid(x)
            # x = self.logsoftmax(x)

        return x

def resnet34(num_classes=1, include_top=True, use_cbam=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, use_cbam=use_cbam)


def resnet50(num_classes=1, include_top=True, use_cbam=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, use_cbam=use_cbam)


def resnet101(num_classes=1, include_top=True, use_cbam=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top, use_cbam=use_cbam)