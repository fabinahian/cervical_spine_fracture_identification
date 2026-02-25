"""
Stage 4 — Per-vertebra VOI extraction.

Extrudes the 2-D sagittal and coronal segmentation masks into 3-D,
intersects them to isolate each vertebra, and cuts tight bounding-box
crops of the VOI volume for C1–C7.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["extract_vertebra_vois"]

NUM_VERTEBRAE   = 7
VERTEBRA_LABELS = [f"C{i + 1}" for i in range(NUM_VERTEBRAE)]


def _tight_crop_3d(arr: np.ndarray) -> np.ndarray:
    """Return the minimal bounding box enclosing all non-zero voxels.

    Args:
        arr: 3-D array of shape ``(H, D, W)``.

    Returns:
        Tightly cropped sub-array.

    Raises:
        ValueError: If *arr* contains no non-zero voxels.
    """
    nz = np.nonzero(arr)
    if any(len(idx) == 0 for idx in nz):
        raise ValueError("Mask is entirely zero — no vertebra region found.")
    lo = np.min(nz, axis=1)
    hi = np.max(nz, axis=1)
    return arr[lo[0]: hi[0] + 1, lo[1]: hi[1] + 1, lo[2]: hi[2] + 1]


def _extrude_masks(
    sag_mask: np.ndarray,
    cor_mask: np.ndarray,
    H: int, D: int, W: int,
) -> np.ndarray:
    """Extrude 2-D projection masks to 3-D and return their intersection.

    Args:
        sag_mask: Sagittal binary mask of shape ``(H, D, K)``.
        cor_mask: Coronal  binary mask of shape ``(H, W, K)``.
        H, D, W: Spatial dimensions of the VOI volume.

    Returns:
        Intersection array of shape ``(K, H, D, W)`` with values in ``{0, 1}``.
    """
    K = NUM_VERTEBRAE
    ext_sag = np.stack(
        [np.repeat(sag_mask[:, :, i, np.newaxis], W, axis=2) for i in range(K)],
        axis=0,
    )
    ext_cor = np.stack(
        [np.transpose(np.repeat(cor_mask[:, :, i, np.newaxis], D, axis=2), (0, 2, 1))
         for i in range(K)],
        axis=0,
    )
    return ext_sag * ext_cor


def extract_vertebra_vois(
    voi: np.ndarray,
    seg_masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply 3-D masks to the VOI and extract per-vertebra tight crops.

    Args:
        voi: Cervical VOI volume of shape ``(H, D, W)``.
        seg_masks: Segmentation masks from Stage 3.

    Returns:
        Dictionary mapping vertebra label (``"C1"``–``"C7"``) to a cropped
        3-D array.  Vertebrae with empty masks are skipped with a warning.
    """
    H, D, W = voi.shape
    sag_mask = seg_masks["sagittal"]
    cor_mask = seg_masks["coronal"]

    assert sag_mask.shape[:2] == (H, D), \
        f"Sagittal mask {sag_mask.shape[:2]} ≠ VOI (H,D)=({H},{D})"
    assert cor_mask.shape[:2] == (H, W), \
        f"Coronal mask  {cor_mask.shape[:2]} ≠ VOI (H,W)=({H},{W})"

    intersection = _extrude_masks(sag_mask, cor_mask, H, D, W)

    vertebra_vois: dict[str, np.ndarray] = {}
    for i, label in enumerate(VERTEBRA_LABELS):
        try:
            roi = _tight_crop_3d(voi * intersection[i])
            vertebra_vois[label] = roi
            logger.info("  %s: shape=%s", label, roi.shape)
        except ValueError as exc:
            logger.warning("  %s skipped — %s", label, exc)

    return vertebra_vois