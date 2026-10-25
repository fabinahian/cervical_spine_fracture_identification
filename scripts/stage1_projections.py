# path: scripts/stage1_projections.py

"""
Stage 1 — Variance projections.

Generates the 2-D sagittal, coronal, and axial variance projections used for
cervical-spine detection, following the cascaded slice selection used to build
the training data:

  1. sagittal: slices ``SAGITTAL_SLICE_RANGE``;
  2. coronal:  slices spanned by the sagittal detection's horizontal extent;
  3. axial:    slices between the mean top and mean bottom of the sagittal and
               coronal detections.

Within the selected range every slice is CLAHE-enhanced; slices outside it are
left at zero and still enter the variance. The projection is then enhanced
with CLAHE and min-max rescaled.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage import exposure

from config import SAGITTAL_SLICE_RANGE, SLICE_CLAHE
from utils import normalize_slice

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


def _variance_projection(volume: np.ndarray, view: str, slice_range: tuple[int, int]) -> np.ndarray:
    """Variance projection of *volume* along *view* over *slice_range*.

    Args:
        volume: Min-max normalised volume of shape ``(D, H, W)``.
        view: Projection view.
        slice_range: ``(start, stop)`` slice indices along the view axis.

    Returns:
        2-D float64 projection in ``[0, 1]``.
    """
    oriented = _reorient(volume, view)
    n = oriented.shape[0]
    start, stop = min(max(0, slice_range[0]), n), min(slice_range[1], n)

    slices = np.zeros_like(oriented, dtype=np.float64)
    for i in range(start, stop):
        slices[i] = exposure.equalize_adapthist(oriented[i]) if SLICE_CLAHE else oriented[i]

    proj = exposure.equalize_adapthist(np.var(slices, axis=0))
    return normalize_slice(proj)


def generate_projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the cascaded sagittal → coronal → axial variance projections.

    The coronal and axial slice ranges are derived from YOLO detections on the
    preceding views, so the detectors of Stage 2 are run here as well.

    Args:
        volume: Windowed CT volume of shape ``(D, H, W)``.

    Returns:
        Dictionary mapping view name → 2-D float64 projection in ``[0, 1]``.
    """
    from scripts.stage2_voi import detect_view   # local import: Stage 2 owns the detectors

    vol = volume.astype(np.float64)
    vol = (vol - vol.min()) / max(vol.max() - vol.min(), 1e-12)

    projections: dict[str, np.ndarray] = {}
    projections["sagittal"] = _variance_projection(vol, "sagittal", SAGITTAL_SLICE_RANGE)
    s_y0, s_y1, s_x0, s_x1 = detect_view(projections["sagittal"], "sagittal")

    projections["coronal"] = _variance_projection(vol, "coronal", (s_x0, s_x1 + 1))
    c_y0, c_y1, _, _ = detect_view(projections["coronal"], "coronal")

    axial_range = (int((s_y0 + c_y0) / 2), int((s_y1 + c_y1) / 2))
    projections["axial"] = _variance_projection(vol, "axial", axial_range)

    for view, proj in projections.items():
        logger.debug("Generated %s projection: shape=%s", view, proj.shape)
    logger.info("  coronal slices %s, axial slices %s", (s_x0, s_x1 + 1), axial_range)
    return projections
