"""Simple Fully Convolutional Network (SFCN) for brain-age regression.

Peng et al., MedIA 2021. Adapted for scalar regression (single linear head
instead of the original soft-label classification over age bins).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_c: int, out_c: int, pool: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_c),
    ]
    if pool:
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class SFCN(nn.Module):
    """SFCN regressor.

    Channel progression follows Peng et al.: 32, 64, 128, 256, 256, 64.
    The last conv block uses AvgPool instead of MaxPool to aggregate the
    spatial dim into a single feature vector before the regression head.
    """

    def __init__(self, in_channels: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.Sequential(
                nn.Conv3d(256, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
                nn.AvgPool3d(kernel_size=(5, 6, 5)),  # for [160,192,160] / 2^5
                nn.ReLU(inplace=True),
            ),
        )
        self.dropout = nn.Dropout3d(dropout)
        self.head = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x.flatten(1).mean(dim=1)  # global avg over any leftover spatial dim


def build_sfcn(in_channels: int = 1, dropout: float = 0.5) -> SFCN:
    return SFCN(in_channels=in_channels, dropout=dropout)


class SFCNClassifier(nn.Module):
    """SFCN backbone with a classification head — adherence predictors for the
    categorical conditioning attributes (sex; dx = AD/MCI/CN or 4-way).

    Shares Peng et al.'s conv progression with :class:`SFCN`, but replaces the
    fixed-kernel AvgPool + scalar head with ``AdaptiveAvgPool3d(1)`` (so it is
    robust to the generated-volume resolution) + a linear logits head. The same
    module covers binary (sex, num_classes=2) and multi-class (dx) targets.
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
            _conv_block(256, 256),
            nn.Sequential(
                nn.Conv3d(256, 64, kernel_size=1, bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
            ),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)   # (B, 64)
        x = self.dropout(x)
        return self.head(x)            # (B, num_classes) logits


def build_sfcn_classifier(num_classes: int, in_channels: int = 1,
                          dropout: float = 0.5) -> SFCNClassifier:
    return SFCNClassifier(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
