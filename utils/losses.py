# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialKLLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for Evidential Deep Learning.
    Penalizes regions with high uncertainty that diverge from the Dirichlet prior.
    """

    def __init__(self, annealing_step=10):
        super().__init__()
        self.annealing_step = annealing_step

    def forward(self, alpha, target, global_step):
        # Dynamic annealing coefficient, to prevent the network from collapsing prematurely due to excessive KL penalty at an early stage
        annealing_coef = min(1.0, global_step / self.annealing_step)

        # Convert the "target" into the required format for the "evidence" space
        S = torch.sum(alpha, dim=1, keepdim=True)
        # Calculate the Dirichlet KL divergence (a simplified version of the log-likelihood)
        loss = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        return torch.mean(loss) * annealing_coef


class EdgeAwareLoss(nn.Module):
    """
    Boundary-aware Loss for BD-Net.
    Dynamically extracts morphological edges from Ground Truth masks using spatial gradients.
    """

    def __init__(self):
        super().__init__()
        # Define the fixed-weight Laplacian operator to extract the true edges of GT
        laplacian_kernel = torch.tensor([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], dtype=torch.float32)
        self.laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        self.bce = nn.BCEWithLogitsLoss()

    def extract_edges(self, mask):
        # Calculate the first or second derivatives of GT in the video memory to extract the edges.
        self.laplacian_kernel = self.laplacian_kernel.to(mask.device)
        edge = F.conv2d(mask, self.laplacian_kernel, padding=1)
        edge = torch.clamp(torch.abs(edge), 0, 1)  # 二值化边界
        return edge

    def forward(self, edge_logits, target_mask):
        # Dynamically extract the GT edges
        gt_edges = self.extract_edges(target_mask)
        # Computing the binary cross-entropy of the boundary
        edge_loss = self.bce(edge_logits, gt_edges)
        return edge_loss


class MDFTotalLoss(nn.Module):
    """
    Composite Loss Function for MDF-Net.
    L_total = L_region(Dice+BCE) + lambda_1 * L_edge + lambda_2 * L_edl
    """

    def __init__(self, lambda_edge=0.5, lambda_edl=0.1):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.lambda_edl = lambda_edl
        self.edge_loss = EdgeAwareLoss()
        self.region_bce = nn.BCEWithLogitsLoss()

    def forward(self, region_logits, edge_logits, alpha_dwi, target_mask, global_step):
        # 1. Macroscopic Region Loss
        l_region = self.region_bce(region_logits, target_mask)

        # 2. Microscopic Edge Loss (BD-Net supervision)
        l_edge = self.edge_loss(edge_logits, target_mask)

        # 3. (Optional) EDL Regularization loss could be added here using alpha_dwi

        return l_region + self.lambda_edge * l_edge