# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialFusion(nn.Module):
    """
    Mismatch-Driven Evidential Fusion Module (EDL-Fusion).
    This module quantifies voxel-wise epistemic uncertainty by parameterizing
    the feature distributions using Dirichlet priors.
    """

    def __init__(self, in_channels, num_classes=2):
        super(EvidentialFusion, self).__init__()
        self.num_classes = num_classes

        self.evidence_extractor_dwi = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.evidence_extractor_flair = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def compute_evidence_and_uncertainty(self, x, extractor):
        """
        math: e = softplus(x), alpha = e + 1, u = K / sum(alpha)
        """
        evidence = F.softplus(extractor(x))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return evidence, alpha, uncertainty

    def forward(self, feat_dwi, feat_flair):
        # 1. Epistemic Uncertainty
        _, _, u_dwi = self.compute_evidence_and_uncertainty(feat_dwi, self.evidence_extractor_dwi)
        _, _, u_flair = self.compute_evidence_and_uncertainty(feat_flair, self.evidence_extractor_flair)

        # 2. Uncertainty-Aware Attention
        # The higher the uncertainty of a region, the lower the feature weight given (suppressing the artifacts of low signal-to-noise ratio FLAIR).
        weight_dwi = torch.exp(-u_dwi)
        weight_flair = torch.exp(-u_flair)

        sum_weights = weight_dwi + weight_flair
        norm_w_dwi = weight_dwi / sum_weights
        norm_w_flair = weight_flair / sum_weights

        # 3. Re-calibration and Aggregation of Evidence Characteristics
        refined_dwi = feat_dwi * norm_w_dwi
        refined_flair = feat_flair * norm_w_flair

        fused_feat = torch.cat([refined_dwi, refined_flair], dim=1)
        out_fused = self.fusion_conv(fused_feat)

        return out_fused, u_dwi, u_flair