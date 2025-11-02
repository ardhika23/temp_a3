import torch
import torch.nn as nn
import torch.nn.functional as F

# Shared building blocks
class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv2d(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up2d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2d(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv2d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 2D UNets
class UNet2DBaseline(nn.Module):
    """
    Really standard 2D UNet suitable for OASIS 2D.
    Use this to get EASY difficulty running quickly.
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv2d(n_channels, 64)
        self.down1 = Down2d(64, 128)
        self.down2 = Down2d(128, 256)
        self.down3 = Down2d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down2d(512, 1024 // factor)
        self.up1 = Up2d(1024, 512 // factor, bilinear)
        self.up2 = Up2d(512, 256 // factor, bilinear)
        self.up3 = Up2d(256, 128 // factor, bilinear)
        self.up4 = Up2d(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet2DHipMRI(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_ch=64):
        super().__init__()
        self.inc = DoubleConv2d(n_channels, base_ch)
        self.down1 = Down2d(base_ch, base_ch * 2)
        self.down2 = Down2d(base_ch * 2, base_ch * 4)
        self.down3 = Down2d(base_ch * 4, base_ch * 8)
        self.down4 = Down2d(base_ch * 8, base_ch * 8)
        self.up1 = Up2d(base_ch * 16, base_ch * 4)
        self.up2 = Up2d(base_ch * 8, base_ch * 2)
        self.up3 = Up2d(base_ch * 4, base_ch)
        self.up4 = Up2d(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# 3D UNets
class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up3d(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3d(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose3d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3DBaseline(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv3d(n_channels, 32)
        self.down1 = Down3d(32, 64)
        self.down2 = Down3d(64, 128)
        self.down3 = Down3d(128, 256)
        self.down4 = Down3d(256, 256)
        self.up1 = Up3d(512, 128)
        self.up2 = Up3d(256, 64)
        self.up3 = Up3d(128, 32)
        self.up4 = Up3d(64, 32)
        self.outc = nn.Conv3d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ContextBlock3d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.BatchNorm3d(ch),
        )

    def forward(self, x):
        out = self.block(x)
        return F.relu(out + x, inplace=True)


class UNet3DImproved(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = nn.Sequential(DoubleConv3d(n_channels, 32), ContextBlock3d(32))
        self.down1 = nn.Sequential(Down3d(32, 64), ContextBlock3d(64))
        self.down2 = nn.Sequential(Down3d(64, 128), ContextBlock3d(128))
        self.down3 = nn.Sequential(Down3d(128, 256), ContextBlock3d(256))
        self.down4 = nn.Sequential(Down3d(256, 256), ContextBlock3d(256))

        self.up1 = Up3d(512, 128)
        self.cb1 = ContextBlock3d(128)
        self.up2 = Up3d(256, 64)
        self.cb2 = ContextBlock3d(64)
        self.up3 = Up3d(128, 32)
        self.cb3 = ContextBlock3d(32)
        self.up4 = Up3d(64, 32)
        self.cb4 = ContextBlock3d(32)
        self.outc = nn.Conv3d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.cb1(x)
        x = self.up2(x, x3)
        x = self.cb2(x)
        x = self.up3(x, x2)
        x = self.cb3(x)
        x = self.up4(x, x1)
        x = self.cb4(x)
        return self.outc(x)

# Helper factory
def build_model(name: str, in_channels=1, out_channels=1):
    name = name.lower()
    if name == "2d":
        return UNet2DBaseline(in_channels, out_channels)
    if name == "2d-hip":
        return UNet2DHipMRI(in_channels, out_channels)
    if name == "3d":
        return UNet3DBaseline(in_channels, out_channels)
    if name == "3d-improved":
        return UNet3DImproved(in_channels, out_channels)
    raise ValueError(f"Unknown model name: {name}")