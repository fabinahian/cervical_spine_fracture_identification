"""
Stage 1 — Variance projections.

Generates 2-D sagittal, coronal, and axial projections from the
windowed CT volume by computing pixel-wise variance across the slice
stack (optionally preceded by per-slice CLAHE), then enhancing the
result with CLAHE.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from skimage import exposure

from config import SAGITTAL_SLICE_RANGE, SLICE_CLAHE
from utils import clahe_volume, normalize_slice

logger = logging.getLogger(__name__)

__all__ = ["generate_projections"]


def _reorient(volume: np.ndarray, view: str) -> np.ndarray:
    """Reorient *volume* so that axis 0 iterates over slices in *view*.

    Args:
        volume: Axial volume of shape ``(D, H, W)``.
        view: ``"sagittal"``, ``"coronal"``, or ``"axial"``.

    Returns:
        Reoriented array with the slice axis first.
    """
    if view == "sagittal":
        return np.transpose(volume, (2, 0, 1))
    if view == "coronal":
        return np.transpose(volume, (1, 0, 2))
    return volume


def generate_projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Compute 2-D variance projections along three anatomical views.

    The sagittal stack is trimmed to ``SAGITTAL_SLICE_RANGE`` before
    projection to focus on the cervical region.

    Args:
        volume: Windowed CT volume of shape ``(D, H, W)``.

    Returns:
        Dictionary mapping view name → 2-D float64 projection in ``[0, 1]``.
    """
    slice_ranges: dict[str, Optional[tuple[int, int]]] = {
        "sagittal": SAGITTAL_SLICE_RANGE,
        "coronal":  None,
        "axial":    None,
    }

    projections: dict[str, np.ndarray] = {}
    for view in ("sagittal", "coronal", "axial"):
        oriented = _reorient(volume, view).astype(np.float64)
        s, e = slice_ranges[view] or (0, oriented.shape[0])
        slices = oriented[s: min(e, oriented.shape[0])]

        if SLICE_CLAHE:
            slices = clahe_volume(slices)

        proj = normalize_slice(np.var(slices, axis=0))
        proj = exposure.equalize_adapthist(proj)
        projections[view] = proj
        logger.debug("Generated %s projection: shape=%s", view, proj.shape)

    return projections