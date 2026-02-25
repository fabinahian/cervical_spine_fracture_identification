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
    vol = apply_window(vol, BONE_WINDOW[0], BONE_WINDOW[1])
    return ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)


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


def _proj_stack_tensor(vol: np.ndarray, device: torch.device) -> torch.Tensor:
    """Preprocess using the *projection-stack* strategy (mirrors PNG dataset).

    Selects ``N_SLICE × 5`` evenly-spaced slices; groups sequentially in 5s.

    Args:
        vol: uint8 array of shape ``(depth, H, W)``.
        device: Target device.

    Returns:
        Float tensor of shape ``(1, N_SLICE, 5, H, W)`` in ``[0, 1]``.
    """
    idx = np.linspace(0, vol.shape[0] - 1, CLS_N_SLICE * CLS_IN_CHANS).astype(int)
    groups = [
        _group_to_tensor(vol[idx[g * CLS_IN_CHANS: (g + 1) * CLS_IN_CHANS]].transpose(1, 2, 0))
        for g in range(CLS_N_SLICE)
    ]
    return torch.tensor(np.stack(groups)).float().unsqueeze(0).to(device)


_PREPROCESSORS = {
    "slice_stack_classification": _slice_stack_tensor,
    "proj_stack_classification":  _proj_stack_tensor,
}


def classify_vertebrae(
    vertebra_vois: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Estimate fracture probability for each vertebra using both classifiers.

    Args:
        vertebra_vois: Per-vertebra volumes from Stage 4.
        device: Torch device for inference.

    Returns:
        Nested dict ``{model_name: {label: probability}}``.
        Missing vertebrae are stored as ``float("nan")``.
    """
    results: dict[str, dict[str, float]] = {}

    for model_name, weight_path in CLS_WEIGHTS.items():
        stem = os.path.splitext(os.path.basename(weight_path))[0]
        preprocess = _PREPROCESSORS[stem]

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