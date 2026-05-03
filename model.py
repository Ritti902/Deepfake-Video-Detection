"""
model.py — DeepfakeDetector
============================
ResNeXt50_32x4d + LSTM architecture for deepfake video detection.

Attribute names match the original training checkpoints:
  self.model      = ResNeXt50 backbone  → state-dict keys: model.0.* … model.7.*
  self.lstm       = LSTM
  self.linear1    = classifier head
  pooling/dropout = functional (no state-dict keys)

build_model() is provided for train.py compatibility.
"""

import os
import re
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepfakeDetector(nn.Module):
    """
    ResNeXt50_32x4d + LSTM deepfake detector.

    Args
    ────
    frames     : Sequence length (number of face frames per clip).
    hidden     : LSTM hidden size. Default 2048.
    classes    : Output classes. Default 2 (REAL / FAKE).
    dropout    : Dropout probability. Default 0.4.
    pretrained : Load ImageNet weights (for training only). Default False.
    """

    def __init__(
        self,
        frames:     int   = 150,
        hidden:     int   = 2048,
        classes:    int   = 2,
        dropout:    float = 0.4,
        pretrained: bool  = False,
    ):
        super().__init__()

        self.frames    = frames
        self.hidden    = hidden
        self.classes   = classes
        self.dropout_p = dropout

        # ── Backbone ─────────────────────────────────────────────────────────
        # Attribute is named `self.model` so state-dict keys are "model.0.*" …
        # matching the original training checkpoints
        weights    = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if pretrained else None
        resnext    = models.resnext50_32x4d(weights=weights)
        self.model = nn.Sequential(*list(resnext.children())[:-2])

        # ── Temporal model ────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = 2048,
            hidden_size = hidden,
            batch_first = True,
        )

        # ── Classifier ───────────────────────────────────────────────────────
        # Named `linear1` to match original training checkpoint key names
        self.linear1 = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x : (batch, frames, 3, H, W)
        Returns:   (batch, classes)
        """
        B, T, C, H, W = x.shape

        x     = x.view(B * T, C, H, W)
        feats = self.model(x)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        feats = feats.view(B * T, -1)
        feats = feats.view(B, T, -1)

        out, _ = self.lstm(feats)
        last   = out[:, -1, :]
        last   = F.dropout(last, p=self.dropout_p, training=self.training)
        return self.linear1(last)

    def extra_repr(self) -> str:
        return (f"frames={self.frames}, hidden={self.hidden}, "
                f"classes={self.classes}, dropout={self.dropout_p}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory — used by train.py
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    frames:     int   = 150,
    hidden:     int   = 2048,
    classes:    int   = 2,
    dropout:    float = 0.4,
    pretrained: bool  = False,
) -> DeepfakeDetector:
    return DeepfakeDetector(
        frames=frames, hidden=hidden,
        classes=classes, dropout=dropout,
        pretrained=pretrained,
    )