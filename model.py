"""
model.py
--------
CNN-Transformer architecture for vertebra-level fracture classification.

The backbone (EfficientNetV2-S) independently encodes each of the N_SLICE
groups of IN_CHANS adjacent sagittal slices. The resulting feature sequence
is refined by a 2-layer Transformer Encoder with sinusoidal positional
encoding, then decoded by an MLP head into per-group fracture logits.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn

from config import (
    CLS_BACKBONE, CLS_DROP_RATE, CLS_DROP_RATE_LAST,
    CLS_IMAGE_SIZE, CLS_IN_CHANS, CLS_N_SLICE, CLS_OUT_DIM,
    CLS_TRANS_LAYERS, CLS_TRANS_NHEAD,
)

__all__ = ["VertebralClassifier"]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequence inputs.

    Args:
        d_model: Dimensionality of the embedding space.
        max_len: Maximum supported sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe  = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* (batch, seq_len, d_model)."""
        return x + self.pe[:, : x.size(1)]


class VertebralClassifier(nn.Module):
    """CNN-Transformer for vertebra-level fracture probability estimation.

    Input:  ``(batch, N_SLICE, IN_CHANS, H, W)``
    Output: ``(batch, N_SLICE)`` raw logits (apply sigmoid for probabilities)
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            CLS_BACKBONE, in_chans=CLS_IN_CHANS, num_classes=CLS_OUT_DIM,
            features_only=False, drop_rate=CLS_DROP_RATE, pretrained=False,
        )
        hdim = self._strip_head(self.encoder)

        if hdim % CLS_TRANS_NHEAD != 0:
            raise ValueError(
                f"Feature dim {hdim} not divisible by CLS_TRANS_NHEAD={CLS_TRANS_NHEAD}."
            )

        self.pos_encoder = PositionalEncoding(hdim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hdim, nhead=CLS_TRANS_NHEAD,
                dropout=CLS_DROP_RATE, batch_first=True,
            ),
            num_layers=CLS_TRANS_LAYERS,
        )
        self.head = nn.Sequential(
            nn.Linear(hdim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(CLS_DROP_RATE_LAST),
            nn.LeakyReLU(0.1),
            nn.Linear(256, CLS_OUT_DIM),
        )

    @staticmethod
    def _strip_head(encoder: nn.Module) -> int:
        """Remove the classification head and return the feature dimension."""
        name = CLS_BACKBONE.lower()
        if "efficient" in name:
            hdim = encoder.conv_head.out_channels
            encoder.classifier = nn.Identity()
        elif "convnext" in name:
            hdim = encoder.head.fc.in_features
            encoder.head.fc = nn.Identity()
        elif "deit3" in name:
            hdim = encoder.head.in_features
            encoder.head = nn.Identity()
        else:
            hdim = encoder.get_classifier().in_features
            encoder.reset_classifier(0)
        return hdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x    = x.view(bs * CLS_N_SLICE, CLS_IN_CHANS, CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)
        feat = self.encoder(x).view(bs, CLS_N_SLICE, -1)
        feat = self.transformer(self.pos_encoder(feat))
        return self.head(feat.contiguous().view(bs * CLS_N_SLICE, -1)).view(bs, CLS_N_SLICE)