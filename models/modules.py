# coding=utf8
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class DualStreamEncoder(nn.Module):
    """
    Dual-stream Encoder to prevent premature semantic dilution across DWI and FLAIR.
    """

    def __init__(self, in_channels=1, base_c=32):
        super().__init__()
        # DWI Stream
        self.dwi_enc1 = ConvBlock(in_channels, base_c)
        self.dwi_enc2 = ConvBlock(base_c, base_c * 2)
        self.dwi_enc3 = ConvBlock(base_c * 2, base_c * 4)

        # FLAIR Stream
        self.flair_enc1 = ConvBlock(in_channels, base_c)
        self.flair_enc2 = ConvBlock(base_c, base_c * 2)
        self.flair_enc3 = ConvBlock(base_c * 2, base_c * 4)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, dwi, flair):
        # Stage 1
        d1 = self.dwi_enc1(dwi)
        f1 = self.flair_enc1(flair)
        # Stage 2
        d2 = self.dwi_enc2(self.pool(d1))
        f2 = self.flair_enc2(self.pool(f1))
        # Stage 3
        d3 = self.dwi_enc3(self.pool(d2))
        f3 = self.flair_enc3(self.pool(f2))

        return (d1, d2, d3), (f1, f2, f3)


class BDNetDecoder(nn.Module):
    """
    Boundary-Aware Task Decoupling Module (BD-Net).
    Explicitly resolves optimization conflict between region and edge extraction.
    """

    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        # Macroscopic Region Stream
        self.region_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

        # Microscopic Edge Localization Stream
        self.boundary_decoder = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        )

        # Boundary-to-Region Rectification
        self.rectification_conv = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)

    def forward(self, fused_features):
        # 1. regional prediction
        region_logits = self.region_decoder(fused_features)

        # 2. Boundary Prediction (High-Frequency Features)
        edge_logits = self.boundary_decoder(fused_features)

        # 3. Injecting edge features into regional features to refine the topological structure of the contour
        combined = torch.cat([region_logits, edge_logits], dim=1)
        final_region_logits = self.rectification_conv(combined)

        return final_region_logits, edge_logits