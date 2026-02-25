"""
Stage 3 — Multilabel U-Net segmentation.

Pads and resizes each VOI projection to the model's input resolution,
runs sigmoid + threshold inference, then inverts the padding/resize to
return binary masks aligned to the original projection grid.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import (
    SEG_IN_CHANNELS, SEG_INPUT_SIZE, SEG_MEAN,
    SEG_NUM_CLASSES, SEG_STD, SEG_THRESHOLD, SEG_WEIGHTS,
)

logger = logging.getLogger(__name__)

__all__ = ["segment_voi"]


def _preprocess(
    proj: np.ndarray,
    view: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad to square, resize to ``SEG_INPUT_SIZE``, and normalise.

    Args:
        proj: 2-D float ``[0, 1]`` projection of shape ``(H, W)``.
        view: View name used to select normalisation statistics.

    Returns:
        ``(tensor, original_hw)`` — input tensor ``(1, C, 256, 256)`` and
        the original ``(H, W)`` needed to invert the transform.
    """
    original_hw = proj.shape[:2]
    image = Image.fromarray((proj * 255).astype(np.uint8), mode="L")

    w, h = image.size
    size = max(w, h)
    padded = Image.new("L", (size, size), color=0)
    padded.paste(image, ((size - w) // 2, (size - h) // 2))
    image = padded.resize(SEG_INPUT_SIZE, Image.LANCZOS)
    if SEG_IN_CHANNELS != 1:
        image = image.convert("RGB")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=SEG_MEAN[view], std=SEG_STD[view]),
    ])
    return tfm(image).unsqueeze(0), original_hw


def _unprocess(
    mask: np.ndarray,
    original_hw: tuple[int, int],
) -> np.ndarray:
    """Invert pad-and-resize to align the mask with the original projection.

    Uses nearest-neighbour interpolation to preserve binary class boundaries.

    Args:
        mask: Predicted binary mask of shape ``(256, 256, C)``.
        original_hw: ``(H, W)`` of the projection before pre-processing.

    Returns:
        Mask of shape ``(H, W, C)``.
    """
    orig_h, orig_w = original_hw
    max_dim = max(orig_h, orig_w)

    scaled = np.stack(
        [np.array(Image.fromarray(mask[:, :, c]).resize(
            (max_dim, max_dim), Image.NEAREST))
         for c in range(mask.shape[2])],
        axis=2,
    )

    if orig_w >= orig_h:
        pad = (max_dim - orig_h) // 2
        scaled = scaled[pad: pad + orig_h, :, :]
    else:
        pad = (max_dim - orig_w) // 2
        scaled = scaled[:, pad: pad + orig_w, :]

    return np.stack(
        [np.array(Image.fromarray(scaled[:, :, c]).resize(
            (orig_w, orig_h), Image.NEAREST))
         for c in range(scaled.shape[2])],
        axis=2,
    )


def segment_voi(
    voi_projections: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run multilabel segmentation on the VOI projections.

    Args:
        voi_projections: Energy projections from Stage 2.
        device: Torch device for inference.

    Returns:
        Dictionary mapping ``"sagittal"`` / ``"coronal"`` → binary mask
        array of shape ``(H, W, SEG_NUM_CLASSES)`` with values in ``{0, 1}``.
    """
    seg_masks: dict[str, np.ndarray] = {}

    for view in ("sagittal", "coronal"):
        tensor, original_hw = _preprocess(voi_projections[view], view)

        checkpoint = torch.load(SEG_WEIGHTS[view], map_location=device)
        model = checkpoint["model"].to(device).eval()

        with torch.no_grad():
            pred = (torch.sigmoid(model(tensor.to(device))) > SEG_THRESHOLD).float()

        mask = np.transpose(pred.squeeze(0).cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        seg_masks[view] = _unprocess(mask, original_hw)
        logger.info("  %s mask: shape=%s  pos_px=%d",
                    view, seg_masks[view].shape, seg_masks[view].sum())

        del model, checkpoint
        torch.cuda.empty_cache()

    return seg_masks