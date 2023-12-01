import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n=4):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n

        self.double_conv_1 = DoubleConv(self.in_channels, self.n)

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv_2 = DoubleConv(self.n, 2*self.n)

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.double_conv_3 = DoubleConv(2*self.n, 4*self.n)

        self.deconv_1 = nn.ConvTranspose2d(4*self.n, 2*self.n, kernel_size=2, stride=2)

        self.double_conv_4 = DoubleConv(4*self.n, 2*self.n)

        self.deconv_2 = nn.ConvTranspose2d(2*self.n, self.n, kernel_size=2, stride=2)

        self.double_conv_5 = DoubleConv(2*self.n, self.n)

        self.final_conv = nn.Conv2d(self.n, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        out1 = self.double_conv_1(x)

        out2 = self.pool_1(out1)
        out2 = self.double_conv_2(out2)

        out3 = self.pool_2(out2)
        out3 = self.double_conv_3(out3)
        out3 = self.deconv_1(out3)

        out4 = torch.cat([out2, out3], dim=1)
        out4 = self.double_conv_4(out4)
        out4 = self.deconv_2(out4)

        out5 = torch.cat([out1, out4], dim=1)
        out5 = self.double_conv_5(out5)
        final_out = self.final_conv(out5)

        return final_out


