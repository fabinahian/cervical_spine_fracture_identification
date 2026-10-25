"""
seg3d_pipeline/stages_3d.py
---------------------------
Stages 3–4 of the comparison arm.

Stage 3 (3D) — the windowed spine VOI is scaled to ``[0, 1]`` with the fixed
bone window, resized to 128³, segmented by the five fold models (sigmoid
probabilities averaged), resized back to the VOI grid one channel at a time,
and collapsed to a label map (most probable vertebra where its probability
exceeds 0.5).

Stage 4 (3D) — each vertebra is cut out with the tight bounding box of its
label; voxels inside the box that do not carry the label are set to air.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Resize

from config import BONE_WINDOW
from seg3d_pipeline.config_3d import (
    BACKGROUND_HU, SEG3D_BACKBONE, SEG3D_INPUT_SIZE, SEG3D_THRESHOLD, SEG3D_WEIGHTS,
)
from seg3d_pipeline.model_3d import build_seg3d_model

logger = logging.getLogger(__name__)

__all__ = ["segment_voi_3d", "extract_vertebra_vois_3d"]

VERTEBRA_LABELS = [f"C{i + 1}" for i in range(7)]


def _preprocess(voi: np.ndarray) -> torch.Tensor:
    lo, hi = BONE_WINDOW[0] - BONE_WINDOW[1] / 2, BONE_WINDOW[0] + BONE_WINDOW[1] / 2
    vol = (np.clip(voi.astype(np.float32), lo, hi) - lo) / (hi - lo)
    image = (np.repeat(vol[None], 3, axis=0) * 255).astype(np.uint8)          # (3, H, D, W)
    image = Resize(list(SEG3D_INPUT_SIZE))(image)
    image = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


@torch.no_grad()
def segment_voi_3d(voi: np.ndarray, device: torch.device) -> np.ndarray:
    """Return a uint8 label map (0 = background, 1–7 = C1–C7) on the VOI grid."""
    x = _preprocess(voi).to(device)
    probs = None
    for path in SEG3D_WEIGHTS:
        net = build_seg3d_model(SEG3D_BACKBONE)
        net.load_state_dict(torch.load(path, map_location=device))
        net = net.to(device).eval()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = net(x).sigmoid().float()
        probs = p if probs is None else probs + p
        del net
    probs /= len(SEG3D_WEIGHTS)

    best, label = None, None
    for c in range(probs.shape[1]):
        chan = F.interpolate(probs[:, c:c + 1], size=voi.shape, mode="trilinear", align_corners=False)[0, 0]
        if best is None:
            best, label = chan, torch.zeros_like(chan, dtype=torch.int64)
        else:
            better = chan > best
            best = torch.where(better, chan, best)
            label = torch.where(better, torch.full_like(label, c), label)
    labelmap = torch.where(best > SEG3D_THRESHOLD, label + 1, torch.zeros_like(label)).to(torch.uint8).cpu().numpy()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("  3D label map: shape=%s  labelled voxels=%d", labelmap.shape, int((labelmap > 0).sum()))
    return labelmap


def extract_vertebra_vois_3d(voi: np.ndarray, labelmap: np.ndarray) -> dict[str, np.ndarray]:
    """Tight per-vertebra crops with non-vertebra voxels set to ``BACKGROUND_HU``."""
    out: dict[str, np.ndarray] = {}
    for v, name in enumerate(VERTEBRA_LABELS, start=1):
        sel = labelmap == v
        idx = np.nonzero(sel)
        if len(idx[0]) == 0:
            logger.warning("  %s skipped — empty 3D label", name)
            continue
        lo = [int(i.min()) for i in idx]
        hi = [int(i.max()) + 1 for i in idx]
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        crop = voi[box].astype(np.float32, copy=True)
        crop[~sel[box]] = BACKGROUND_HU
        out[name] = crop
        logger.info("  %s: shape=%s", name, crop.shape)
    return out
