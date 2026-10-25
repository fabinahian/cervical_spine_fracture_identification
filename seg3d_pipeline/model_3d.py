"""
seg3d_pipeline/model_3d.py
--------------------------
3D U-Net used for the comparison arm: a 2D timm encoder / segmentation-models
U-Net decoder converted to 3D layer by layer (as in the RSNA 2022 winning
solution).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from timm.layers.conv2d_same import Conv2dSame

from seg3d_pipeline.config_3d import SEG3D_N_BLOCKS, SEG3D_OUT_DIM
from seg3d_pipeline.conv3d_same import Conv3dSame

__all__ = ["build_seg3d_model"]


class TimmSegModel(nn.Module):
    """2D U-Net (timm encoder + U-Net decoder) before 3D conversion."""

    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone, in_chans=3, features_only=True,
                                         drop_rate=0.0, drop_path_rate=0.0, pretrained=False)
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [f.shape[1] for f in g]
        decoder_channels = [256, 128, 64, 32, 16]
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels[:SEG3D_N_BLOCKS + 1],
            decoder_channels=decoder_channels[:SEG3D_N_BLOCKS],
            n_blocks=SEG3D_N_BLOCKS,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[SEG3D_N_BLOCKS - 1], SEG3D_OUT_DIM,
                                           kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [0] + self.encoder(x)[:SEG3D_N_BLOCKS]
        return self.segmentation_head(self.decoder(*features))


def _convert_3d(module: nn.Module) -> nn.Module:
    """Replace every 2D layer by its 3D counterpart (weights repeated along depth)."""
    out = module
    if isinstance(module, nn.BatchNorm2d):
        out = nn.BatchNorm3d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                out.weight = module.weight
                out.bias = module.bias
        out.running_mean = module.running_mean
        out.running_var = module.running_var
        out.num_batches_tracked = module.num_batches_tracked
    elif isinstance(module, Conv2dSame):
        out = Conv3dSame(module.in_channels, module.out_channels, kernel_size=module.kernel_size[0],
                         stride=module.stride[0], padding=module.padding[0], dilation=module.dilation[0],
                         groups=module.groups, bias=module.bias is not None)
        out.weight = nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))
    elif isinstance(module, nn.Conv2d):
        out = nn.Conv3d(module.in_channels, module.out_channels, kernel_size=module.kernel_size[0],
                        stride=module.stride[0], padding=module.padding[0], dilation=module.dilation[0],
                        groups=module.groups, bias=module.bias is not None, padding_mode=module.padding_mode)
        out.weight = nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))
    elif isinstance(module, nn.MaxPool2d):
        out = nn.MaxPool3d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                           dilation=module.dilation, ceil_mode=module.ceil_mode)
    elif isinstance(module, nn.AvgPool2d):
        out = nn.AvgPool3d(kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                           ceil_mode=module.ceil_mode)
    for name, child in module.named_children():
        out.add_module(name, _convert_3d(child))
    return out


def build_seg3d_model(backbone: str) -> nn.Module:
    """Return the 3D U-Net with randomly initialised weights."""
    return _convert_3d(TimmSegModel(backbone))
