"""BEV Occupancy Prediction Head for OccSafe-VLA.

Takes fused visual features (16x16 spatial resolution) and predicts
BEV occupancy map (64x64). Auxiliary training signal only —
skipped at inference (zero overhead, cf. OccVLA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVOccupancyHead(nn.Module):
    def __init__(self, in_dim: int, out_resolution: int = 64, num_classes: int = 1,
                 hidden_dim: int = 256, num_patches_per_image: int = 256, norm_type: str = "bn"):
        super().__init__()
        self.num_patches_per_image = num_patches_per_image
        self.spatial_h = self.spatial_w = int(num_patches_per_image ** 0.5)  # 16
        self.out_resolution = out_resolution
        self.num_classes = num_classes

        def _norm(channels):
            if norm_type == "ln":
                return nn.GroupNorm(1, channels)
            return nn.BatchNorm2d(channels)

        self.project = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            _norm(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32->64
            _norm(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, N, D) fused features.
                N >= num_patches_per_image; first view is used.
        Returns:
            logits: (B, num_classes, out_resolution, out_resolution)
        """
        if visual_features.shape[1] > self.num_patches_per_image:
            visual_features = visual_features[:, :self.num_patches_per_image, :]

        B = visual_features.shape[0]
        x = self.project(visual_features)
        x = x.transpose(1, 2).reshape(B, -1, self.spatial_h, self.spatial_w)
        logits = self.decoder(x)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, H, W) binary
        """
        probs = torch.sigmoid(logits.squeeze(1))
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets, reduction='none'
        )
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DepthHead(nn.Module):
    """Monocular depth prediction head, reusing BEVOccupancyHead architecture.

    Input:  (B, N, D) patch features — first 256 tokens (16x16 spatial grid) are used.
    Output: (B, 1, out_resolution, out_resolution) continuous depth map (non-negative via ReLU).

    Unlike BEVOccupancyHead which outputs binary logits, this head outputs
    continuous depth values with SmoothL1 regression loss.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_resolution: int = 64,
                 num_patches_per_image: int = 256):
        super().__init__()
        self.num_patches_per_image = num_patches_per_image
        self.spatial_h = self.spatial_w = int(num_patches_per_image ** 0.5)  # 16
        self.out_resolution = out_resolution

        self.project = nn.Linear(in_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),  # depth is non-negative
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, N, D) fused features.
                N >= num_patches_per_image; first view is used.
        Returns:
            depth: (B, 1, out_resolution, out_resolution) non-negative depth values.
        """
        if visual_features.shape[1] > self.num_patches_per_image:
            visual_features = visual_features[:, :self.num_patches_per_image, :]

        B = visual_features.shape[0]
        x = self.project(visual_features)
        x = x.transpose(1, 2).reshape(B, -1, self.spatial_h, self.spatial_w)
        depth = self.decoder(x)
        return depth


class DepthLoss(nn.Module):
    """SmoothL1 loss for monocular depth prediction.

    More robust to outliers than L1. Expects:
      pred:   (B, 1, H, W) predicted depth
      target: (B, H, W) ground-truth depth
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.unsqueeze(1)  # (B, 1, H, W)
        return self.loss_fn(pred, target)
