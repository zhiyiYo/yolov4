# coding:utf-8
import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    """ Mish 激活函数 """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class CBMBlock(nn.Module):
    """ CBM 块 """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        """ 前馈

        Parameters
        ----------
        x: Tensor of shape `(N, in_channels, H, W)`
            输入

        Returns
        -------
        y: Tensor of shape `(N, out_channels, H, W)`
            输出
        """
        return self.mish(self.bn(self.conv(x)))


class ResidualUnit(nn.Module):
    """ 残差单元 """

    def __init__(self, in_channels, hidden_channels=None):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        hidden_channels: int
            第一个 CBM 块的输出通道数
        """
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.block = nn.Sequential(
            CBMBlock(in_channels, hidden_channels, 1),
            CBMBlock(hidden_channels, in_channels, 3),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualBlock(nn.Module):
    """ 残差块 """

    def __init__(self, in_channels, out_channels, n_blocks):
        """
        Parameters
        ----------
        in_channels: int
            输入通道数

        out_channels: int
            输出通道数

        n_blocks: int
            内部残差单元个数
        """
        super().__init__()
        self.downsample_conv = CBMBlock(
            in_channels, out_channels, 3, stride=2)

        if n_blocks == 1:
            self.split_conv0 = CBMBlock(out_channels, out_channels, 1)
            self.split_conv1 = CBMBlock(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                ResidualUnit(out_channels, out_channels//2),
                CBMBlock(out_channels, out_channels, 1)
            )
            self.concat_conv = CBMBlock(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = CBMBlock(out_channels, out_channels//2, 1)
            self.split_conv1 = CBMBlock(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[ResidualUnit(out_channels//2) for _ in range(n_blocks)],
                CBMBlock(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = CBMBlock(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        # 残差部分
        x0 = self.split_conv0(x)

        # 主干部分
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        return self.concat_conv(x)


class CSPDarkNet(nn.Module):
    """ CSPDarkNet 主干网络 """

    def __init__(self) -> None:
        super().__init__()
        layers = [1, 2, 8, 8, 4]
        channels = [32, 64, 128, 256, 512, 1024]
        self.conv1 = CBMBlock(3, 32, 3)
        self.stages = nn.ModuleList([
            ResidualBlock(channels[i], channels[i+1], layers[i]) for i in range(5)
        ])

    def forward(self, x):
        """ 前馈

        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入

        Returns
        -------
        x3: Tensor of shape `(N, 256, H/8, W/8)`
        x4: Tensor of shape `(N, 512, H/16, W/16)`
        x5: Tensor of shape `(N, 1024, H/32, W/32)`
        """
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)

        # 输出三个特征图
        x3 = self.stages[2](x)
        x4 = self.stages[3](x3)
        x5 = self.stages[4](x4)

        return x3, x4, x5


class CBLBlock(nn.Module):
    """ CBL 块 """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size-1)//2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SPPBlock(nn.Module):
    """ SPP 块 """

    def __init__(self, sizes=(5, 9, 13)):
        super().__init__()
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(size, 1, size//2) for size in sizes
        ])

    def forward(self, x):
        x1 = [pool(x) for pool in self.maxpools[::-1]]
        x1.append(x)
        return torch.cat(x1, dim=1)


class Upsample(nn.Module):
    """ 上采样 """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            CBLBlock(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    def forward(self, x):
        return self.upsample(x)


def make_three_cbl(channels: list):
    """ 创建三个 CBL 块 """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
    )


def make_five_cbl(channels: list):
    """ 创建五个 CBL 块 """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
        CBLBlock(channels[1], channels[2], 3),
        CBLBlock(channels[2], channels[1], 1),
    )


def make_yolo_head(channels: list):
    """ 创建 Yolo 头 """
    return nn.Sequential(
        CBLBlock(channels[0], channels[1], 3),
        nn.Conv2d(channels[1], channels[2], 1),
    )


class Yolo(nn.Module):
    """ Yolov4 神经网络 """

    def __init__(self, n_classes):
        super().__init__()
        # 主干网络
        self.backbone = CSPDarkNet()

        self.conv1 = make_three_cbl([1024, 512, 1024])
        self.SPP = SPPBlock()
        self.conv2 = make_three_cbl([2048, 512, 1024])

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = CBLBlock(512, 256, 1)
        self.make_five_conv1 = make_five_cbl([512, 256, 512])

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = CBLBlock(256, 128, 1)
        self.make_five_conv2 = make_five_cbl([256, 128, 256])

        channel = 3 * (5 + n_classes)
        self.yolo_head3 = make_yolo_head([128, 256, channel])

        self.down_sample1 = CBLBlock(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_cbl([512, 256, 512])

        self.yolo_head2 = make_yolo_head([256, 512, channel])

        self.down_sample2 = CBLBlock(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_cbl([1024, 512, 1024])

        self.yolo_head1 = make_yolo_head([512, 1024, channel])

    def forward(self, x: torch.Tensor):
        """ 前馈

        Parameters
        ----------
        x: Tensor of shape `(N, 3, H, W)`
            输入

        Returns
        -------
        """
        # 主干网络
        x2, x1, x0 = self.backbone(x)

        # (13, 13, 1024) --> (13, 13, 512)
        P5 = self.conv2(self.SPP(self.conv1(x0)))

        # 上采样 P5 之后再拼接, (13, 13, 512) --> (26, 26, 256) --> (26, 26, 512)
        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], dim=1)
        # (26, 26, 512) --> (26, 26, 256)
        P4 = self.make_five_conv1(P4)

        # 上采样 P4 之后再拼接, (26. 26, 256) --> (52, 52, 128) --> (52, 52, 256)
        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], dim=1)
        # (52, 52, 256) --> (52, 52, 128)
        P3 = self.make_five_conv2(P3)

        # 对 P3 进行下采样再和 P4 拼接, (52, 52, 128) --> (26, 26, 256) --> (26, 26, 512)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], dim=1)
        # (26, 26, 512) --> (26, 26, 256)
        P4 = self.make_five_conv3(P4)

        # 对 P4 进行下采样再和 P5 拼接, (26, 26, 256) --> (13, 13, 512) --> (13, 13, 1024)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], dim=1)
        # (13, 13, 1024) --> (13, 13, 512)
        P5 = self.make_five_conv4(P5)

        # 输出的三个特征图
        y2 = self.yolo_head3(P3)  # (N, 3*(5+n_classes), 52, 52)
        y1 = self.yolo_head2(P4)  # (N, 3*(5+n_classes), 26, 26)
        y0 = self.yolo_head1(P5)  # (N, 3*(5+n_classes), 13, 13)

        return y0, y1, y2
