# path: scripts/stage5_classification.py

"""
Stage 5 — Vertebra-level fracture classification.

Loads each CNN-Transformer checkpoint, pre-processes every vertebra VOI
using the strategy that matches the checkpoint's training dataloader, and
returns the mean sigmoid probability across all N_SLICE groups as the
fracture probability for that vertebra.
"""

from __future__ import annotations

import logging
import os

import albumentations
import numpy as np
import torch

from config import (
    BONE_WINDOW, CLS_IMAGE_SIZE, CLS_IN_CHANS, CLS_N_SLICE,
    CLS_PAD, CLS_SLICE_VIEW, CLS_WEIGHTS,
)
from model import VertebralClassifier
from utils import apply_window, pad_numpy_to_square

logger = logging.getLogger(__name__)

__all__ = ["classify_vertebrae"]

NUM_VERTEBRAE   = 7
VERTEBRA_LABELS = [f"C{i + 1}" for i in range(NUM_VERTEBRAE)]

_resize = albumentations.Compose([albumentations.Resize(CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)])


def _prepare_vol(vol: np.ndarray) -> np.ndarray:
    """Reorient, window, and uint8-normalise a vertebra VOI.

    Args:
        vol: Raw VOI of shape ``(H, D, W)``.

    Returns:
        uint8 array of shape ``(depth, H, W)`` ready for slice sampling.
    """
    if CLS_SLICE_VIEW == "sagittal":
        vol = np.transpose(vol, (2, 0, 1))
    elif CLS_SLICE_VIEW == "coronal":
        vol = np.transpose(vol, (1, 0, 2))
    vol = apply_window(vol, BONE_WINDOW[0], BONE_WINDOW[1])
    rng = vol.max() - vol.min()
    if rng == 0:
        return np.zeros(vol.shape, dtype=np.uint8)
    return ((vol - vol.min()) / rng * 255).astype(np.uint8)


def _group_to_tensor(group: np.ndarray) -> np.ndarray:
    """Optionally pad and resize a ``(H, W, 5)`` slice group.

    Args:
        group: Array of shape ``(H, W, 5)``.

    Returns:
        Float32 array of shape ``(5, H, W)`` in ``[0, 1]``.
    """
    if CLS_PAD:
        group = pad_numpy_to_square(group)
    group = _resize(image=group)["image"]
    return group.transpose(2, 0, 1).astype(np.float32) / 255.0


def _slice_stack_tensor(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess using the *slice-stack* strategy (mirrors ``CLSDataset``).

    Selects ``N_SLICE`` evenly-spaced centres in ``[2, depth-3]``;
    each centre yields a 5-slice group via ±2 neighbours.

    Args:
        vol: uint8 array of shape ``(depth, H, W)``.
        device: Target device.

    Returns:
        Float tensor of shape ``(1, N_SLICE, 5, H, W)`` in ``[0, 1]``.
    """
    centres = np.linspace(2, vol.shape[0] - 3, CLS_N_SLICE).astype(int)
    groups = [_group_to_tensor(vol[i - 2: i + 3].transpose(1, 2, 0)) for i in centres]
    return torch.tensor(np.stack(groups)).float().unsqueeze(0).to(device)


def _mip_image(slab: np.ndarray) -> np.ndarray:
    """Maximum-intensity projection of one 5-slice slab (mirrors the MIP PNG generation).

    The ``(H, W, 5)`` slab is resized, projected by taking the maximum over its
    5 slices, and min-max rescaled to uint8.

    Args:
        slab: uint8 array of shape ``(H, W, 5)``.

    Returns:
        uint8 array of shape ``(CLS_IMAGE_SIZE, CLS_IMAGE_SIZE)``.
    """
    if CLS_PAD:
        slab = pad_numpy_to_square(slab)
    mip = _resize(image=slab)["image"].astype(np.uint8).max(axis=2).astype(np.float64)
    rng = mip.max() - mip.min()
    if rng == 0:
        return np.zeros_like(mip, dtype=np.uint8)
    return ((mip - mip.min()) / rng * 255).astype(np.uint8)


def _proj_stack_tensor(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess using the *projection-stack* (MIP) strategy.

    Selects ``N_SLICE × 5`` evenly-spaced centres in ``[2, depth-3]``; each
    centre's ±2 slab is collapsed to one maximum-intensity projection, and
    consecutive MIPs are grouped in 5s as the 5 input channels.

    Args:
        vol: uint8 array of shape ``(depth, H, W)``.
        device: Target device.

    Returns:
        Float tensor of shape ``(1, N_SLICE, 5, H, W)`` in ``[0, 1]``.
    """
    centres = np.linspace(2, vol.shape[0] - 3, CLS_N_SLICE * CLS_IN_CHANS).astype(int)
    mips = [_mip_image(vol[i - 2: i + 3].transpose(1, 2, 0)) for i in centres]
    groups = [
        _resize(image=np.stack(mips[g * CLS_IN_CHANS: (g + 1) * CLS_IN_CHANS], axis=2))["image"]
        .transpose(2, 0, 1).astype(np.float32) / 255.0
        for g in range(CLS_N_SLICE)
    ]
    return torch.tensor(np.stack(groups)).float().unsqueeze(0).to(device)


def _preprocessor(weight_path: str):
    """Pick the preprocessing from the checkpoint name (``slice_stack…`` or ``proj_stack…``)."""
    stem = os.path.splitext(os.path.basename(weight_path))[0]
    if stem.startswith("proj_stack"):
        return stem, _proj_stack_tensor
    if stem.startswith("slice_stack"):
        return stem, _slice_stack_tensor
    raise ValueError(f"Cannot infer preprocessing from checkpoint name '{stem}'.")


def classify_vertebrae(
    vertebra_vois: dict[str, np.ndarray],
    device: torch.device,
    weights: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Estimate fracture probability for each vertebra using both classifiers.

    Args:
        vertebra_vois: Per-vertebra volumes from Stage 4.
        device: Torch device for inference.
        weights: ``{model_name: checkpoint}``; defaults to ``CLS_WEIGHTS``.

    Returns:
        Nested dict ``{model_name: {label: probability}}``.
        Missing vertebrae are stored as ``float("nan")``.
    """
    results: dict[str, dict[str, float]] = {}

    for model_name, weight_path in (weights or CLS_WEIGHTS).items():
        stem, preprocess = _preprocessor(weight_path)

        model = VertebralClassifier().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        logger.info("  Loaded: %s (%s)", model_name, stem)

        probs: dict[str, float] = {}
        for label in VERTEBRA_LABELS:
            if label not in vertebra_vois:
                logger.warning("    %s missing — skipped.", label)
                probs[label] = float("nan")
                continue

            tensor = preprocess(_prepare_vol(vertebra_vois[label]), device)
            with torch.no_grad():
                probs[label] = float(torch.sigmoid(model(tensor)).cpu().numpy().mean())
            logger.info("    %s: %.4f", label, probs[label])

        results[model_name] = probs
        del model
        torch.cuda.empty_cache()

    return results