import torch as T
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_channels = params['in_channels']
        out_channels = params['out_channels']
        hidden_channels = params['hidden_channels']
        kernel_size = params['kernel_size']
        padding = params['padding']
        scale_factor = params['scale_factor']

        self.down1 = DownBlock(in_channels, hidden_channels[0], hidden_channels[1], kernel_size, padding)
        self.down2 = DownBlock(hidden_channels[1], hidden_channels[1], hidden_channels[2], kernel_size, padding)
        self.down3 = DownBlock(hidden_channels[2], hidden_channels[2], hidden_channels[3], kernel_size, padding)
        self.down4 = DownBlock(hidden_channels[3], hidden_channels[3], hidden_channels[4], kernel_size, padding)

        self.bottleneck = DownBlock(hidden_channels[4], hidden_channels[4], hidden_channels[5], kernel_size, padding)

        self.up4 = UpBlock(hidden_channels[5] + hidden_channels[4], hidden_channels[4], hidden_channels[3], kernel_size, padding, scale_factor)
        self.up3 = UpBlock(hidden_channels[3] + hidden_channels[3], hidden_channels[3], hidden_channels[2], kernel_size, padding, scale_factor)
        self.up2 = UpBlock(hidden_channels[2] + hidden_channels[2], hidden_channels[2], hidden_channels[1], kernel_size, padding, scale_factor)
        self.up1 = UpBlock(hidden_channels[1] + hidden_channels[1], hidden_channels[1], hidden_channels[0], kernel_size, padding, scale_factor)

        self.final_conv = nn.Conv3d(hidden_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1, p1 = self.down1(x)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)
        x4, p4 = self.down4(p3)

        bn, _ = self.bottleneck(p4)

        x = self.up4(bn, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.final_conv(x)
        return x
