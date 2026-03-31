# coding=utf8
import torch
import torch.nn as nn
from .modules import DualStreamEncoder, ConvBlock, BDNetDecoder
from .evidential import EvidentialFusion


class MDFNet(nn.Module):
    """
    Mismatch-Driven Evidential Fusion and Task Decoupling Network (MDF-Net).
    Paper: Decoding the DWI-FLAIR Mismatch for Acute Ischemic Stroke.
    """

    def __init__(self, input_channels=1, num_classes=1, base_c=32):
        super(MDFNet, self).__init__()

        # 1. Dual-Stream Encoder
        self.encoder = DualStreamEncoder(in_channels=input_channels, base_c=base_c)

        # 2. Bottleneck & Evidential Fusion
        self.dwi_bottleneck = ConvBlock(base_c * 4, base_c * 8)
        self.flair_bottleneck = ConvBlock(base_c * 4, base_c * 8)
        self.edl_fusion = EvidentialFusion(in_channels=base_c * 8, num_classes=2)

        # 3. Decoder with Skip Connections
        self.upconv = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec_block = ConvBlock(base_c * 4 * 3, base_c * 4)  # Fused + Skip_DWI + Skip_FLAIR

        # 4. BD-Net Decoder
        self.bd_net = BDNetDecoder(in_channels=base_c * 4, num_classes=num_classes)

    def forward(self, x_dwi, x_flair):
        # --- Encoder ---
        d_feats, f_feats = self.encoder(x_dwi, x_flair)

        # --- Bottleneck ---
        d_bot = self.dwi_bottleneck(nn.MaxPool2d(2)(d_feats[2]))
        f_bot = self.flair_bottleneck(nn.MaxPool2d(2)(f_feats[2]))

        # --- Mismatch-Driven Evidential Fusion ---
        fused_bot, u_dwi, u_flair = self.edl_fusion(d_bot, f_bot)

        # --- Decoder & Feature Concatenation ---
        dec_feat = self.upconv(fused_bot)

        skip_fused = torch.cat([dec_feat, d_feats[2], f_feats[2]], dim=1)
        dec_out = self.dec_block(skip_fused)

        # --- Task Decoupling (Region + Boundary) ---
        final_region_logits, boundary_logits = self.bd_net(dec_out)


        if self.training:
            return final_region_logits, boundary_logits, u_dwi, u_flair
        else:
            return torch.sigmoid(final_region_logits)