from __future__ import annotations

import torch
from torch import nn


class ViewEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class VolumeDecoder(nn.Module):
    def __init__(self, feature_dim: int = 256, voxel_size: int = 32) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        base_size = voxel_size // 8
        hidden_channels = 128
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels * base_size * base_size * base_size),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
        )

    def forward(self, feature_vector: torch.Tensor) -> torch.Tensor:
        base_size = self.voxel_size // 8
        volume = self.projection(feature_vector)
        volume = volume.view(feature_vector.shape[0], 128, base_size, base_size, base_size)
        return self.decoder(volume)


class MultiViewReconstructionNet(nn.Module):
    def __init__(self, feature_dim: int = 256, voxel_size: int = 32) -> None:
        super().__init__()
        self.encoder = ViewEncoder(feature_dim=feature_dim)
        self.decoder = VolumeDecoder(feature_dim=feature_dim, voxel_size=voxel_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, channels, height, width = images.shape
        flattened = images.view(batch_size * num_views, channels, height, width)
        per_view_features = self.encoder(flattened)
        per_view_features = per_view_features.view(batch_size, num_views, -1)
        fused_features = per_view_features.mean(dim=1)
        logits = self.decoder(fused_features)
        return torch.sigmoid(logits)


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, feature_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.network(points.transpose(1, 2))
        return torch.max(features, dim=2).values


class PointCloudReconstructionNet(nn.Module):
    def __init__(self, input_dim: int = 6, feature_dim: int = 256, voxel_size: int = 32) -> None:
        super().__init__()
        self.encoder = PointCloudEncoder(input_dim=input_dim, feature_dim=feature_dim)
        self.decoder = VolumeDecoder(feature_dim=feature_dim, voxel_size=voxel_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(points)
        logits = self.decoder(encoded)
        return torch.sigmoid(logits)
