import torch
import torch.nn as nn
from torchsummary import summary


class BasicBlock(nn.Module):
    """
    18-layer和34-layer的残差块，里面是两个3x3的卷积层
    """
    channel_exp = 1  # 通道保持不变

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        初始化ResNet BasicBlock的参数
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param stride: 步长
        :param branch: 是否使用residual connection
        :param downsample: 在残差连接分支中是否使用了1x1卷积
        :param kwargs: 其他参数
        :return:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        residual_out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        # 如果downsample是带有1x1卷积核的，那么就处理一下
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity + residual_out)


class Bottleneck(nn.Module):
    """
    针对50-layer,101-layer和152-layer的残差块的实现
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    channel_exp = 4  # 通道扩大4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        """
        初始化ResNet Bottleneck的参数
        :param in_channel: 输入通道数
        :param out_channel: 输出通道数
        :param stride: 步长
        :param downsample: 残差分支是否采用1x1卷积
        :param groups: 组卷积
        :param width_per_group:每组卷积数量
        :return:
        """
        super(Bottleneck, self).__init__()
        # out_channel分别为64,128,256,512
        width = int(out_channel * (width_per_group / 64.)) * groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # 第一个1x1卷积是用来压缩通道数，减少计算复杂度
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width,
                               kernel_size=3, stride=stride, padding=1, bias=False)  # 第二个3x3卷积，将高和宽减半
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=width * self.channel_exp,
                               kernel_size=1, stride=1, bias=False)  # 第三个1x1卷积，将通道数扩大4倍
        self.bn3 = nn.BatchNorm2d(width * self.channel_exp)

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        residual_out = self.bn3(self.conv3(self.bn2(self.conv2(self.bn1(self.conv1)))))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(identity + residual_out)


class ResNet(nn.Module):
    def __init__(self,
                block,
                blocks_num,
                num_classes=1000,
                include_top=True,
                groups=1,
                width_per_group=64):
        """
        ResNet模型初始化
        :param block: 残差块选择为BasicBlock还是BottleNeck
        :param blocks_num: 残差块数量,list
        :param num_classes: 分类数量
        :param include_top: 是否包含全连接层
        :param groups:
        :param width_per_group:
        :return:
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 第一个7x7大卷积核
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 将高和宽缩小一半,变为56x56x64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # stride=1
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.channel_exp, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, out_channel, block_num, stride=1):
        """
        构造BasicBlock或者Bottleneck
        :param block:
        :param out_channel: 残差结构中第一个卷积层的卷积核个数
        :param block_num:
        :param stride:
        :return:
        """
        downsample = None
        # 构造downsample
        # 如果是18-layer和34-layer的conv2_1，即stride=1，则使用实线的残差结构，即不进入if语句
        # 如果是50,101,152，则conv2_1,conv3_1,conv4_1,conv5_1都需要使用虚线残差，只不过conv2_x只会调整通道数，不会进行图片下采样
        if stride != 1 or self.in_channel != out_channel * block.channel_exp:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.channel_exp, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * block.channel_exp))
        layers = []
        # 添加第一个block，18-layer和34-layer的downsample=None
        layers.append(block(self.in_channel, out_channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = out_channel * block.channel_exp  # 对于18和34，输出channel和输入channel一样
        # 添加剩下的实线部分，不需要1x1卷积,stride=1
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                out_channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    summary(resnet34().cuda(), (3, 224, 224))
