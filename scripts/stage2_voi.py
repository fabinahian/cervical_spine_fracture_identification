# path: scripts/stage2_voi.py

"""
Stage 2 — YOLO detection and Volume of Interest extraction.

Runs a YOLO detector on each anatomical projection to localise the
cervical spine, fuses the three bounding boxes into a 3-D crop, and
generates energy-based sagittal and coronal projections of the VOI for
the segmentation stage.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage import exposure
from ultralytics import YOLO

from config import SLICE_CLAHE, VOI_PADDING, YOLO_WEIGHTS
from utils import clahe_volume, normalize_slice

logger = logging.getLogger(__name__)

__all__ = ["detect_voi"]


def _projection_to_bgr(proj: np.ndarray) -> np.ndarray:
    """Convert a normalised ``[0, 1]`` grayscale projection to BGR uint8.

    Args:
        proj: 2-D float array in ``[0, 1]``.

    Returns:
        uint8 BGR array of shape ``(H, W, 3)``.
    """
    return cv2.cvtColor((proj * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _yolo_bbox(
    projection_bgr: np.ndarray,
    model: YOLO,
) -> tuple[int, int, int, int]:
    """Run YOLO and return the highest-confidence bounding box.

    Args:
        projection_bgr: BGR uint8 array of shape ``(H, W, 3)``.
        model: Loaded YOLO model instance.

    Returns:
        Bounding box as ``(y_min, y_max, x_min, x_max)`` in pixel coords.

    Raises:
        ValueError: If no objects are detected.
    """
    results = model.predict(projection_bgr, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        raise ValueError("YOLO found no detections in the projection.")

    best = boxes[boxes.conf.argmax()]
    x_c, y_c, w, h = best.xywhn[0].tolist()
    ih, iw = projection_bgr.shape[:2]
    x_c *= iw; y_c *= ih; w *= iw; h *= ih
    return (int(y_c - h / 2), int(y_c + h / 2),
            int(x_c - w / 2), int(x_c + w / 2))


def _build_voi(
    bboxes: dict[str, tuple[int, int, int, int]],
    volume: np.ndarray,
) -> np.ndarray:
    """Fuse three-view bounding boxes into a 3-D VOI crop.

    Heights are averaged from sagittal + coronal; depth from sagittal +
    axial; width from coronal + axial.

    Args:
        bboxes: Dictionary mapping view name → ``(y_min, y_max, x_min, x_max)``.
        volume: Full interpolated CT volume of shape ``(D, H, W)``.

    Returns:
        Cropped sub-volume of shape ``(H', D', W')``.
    """
    def avg(a: int, b: int) -> int:
        return (a + b) // 2

    H, D, W = volume.shape
    p = VOI_PADDING
    sag, cor, ax = bboxes["sagittal"], bboxes["coronal"], bboxes["axial"]

    h_min = max(0,     avg(cor[0], sag[0]) - p)
    h_max = min(H - 1, avg(cor[1], sag[1]) + p)
    d_min = max(0,     avg(sag[2], ax[0])  - p)
    d_max = min(D - 1, avg(sag[3], ax[1])  + p)
    w_min = max(0,     avg(cor[2], ax[2])  - p)
    w_max = min(W - 1, avg(cor[3], ax[3])  + p)

    return volume[h_min:h_max, d_min:d_max, w_min:w_max]


def _voi_projections(voi: np.ndarray) -> dict[str, np.ndarray]:
    """Generate CLAHE-enhanced energy projections of the VOI.

    Args:
        voi: Cropped volume of shape ``(H, D, W)``.

    Returns:
        Dictionary mapping ``"sagittal"`` / ``"coronal"`` → 2-D projection.
    """
    oriented = {
        "sagittal": np.transpose(voi, (2, 0, 1)).astype(np.float64),
        "coronal":  np.transpose(voi, (1, 0, 2)).astype(np.float64),
    }
    out: dict[str, np.ndarray] = {}
    for view, slices in oriented.items():
        if SLICE_CLAHE:
            slices = clahe_volume(slices)
        energy = normalize_slice(np.sum(slices ** 2, axis=0))
        out[view] = exposure.equalize_adapthist(normalize_slice(energy))
    return out


def detect_voi(
    projections: dict[str, np.ndarray],
    volume: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Detect the cervical VOI with YOLO and return the cropped volume.

    Args:
        projections: Variance projections from Stage 1.
        volume: Interpolated, windowed CT volume of shape ``(D, H, W)``.

    Returns:
        ``(voi, voi_projections)`` where ``voi`` is the cropped sub-volume
        and ``voi_projections`` are energy projections ready for Stage 3.
    """
    models = {view: YOLO(w) for view, w in YOLO_WEIGHTS.items()}

    bboxes: dict[str, tuple[int, int, int, int]] = {}
    for view in ("sagittal", "coronal", "axial"):
        bboxes[view] = _yolo_bbox(_projection_to_bgr(projections[view]), models[view])
        logger.info("  %s bbox: %s", view, bboxes[view])

    voi = _build_voi(bboxes, volume)
    logger.info("  VOI shape: %s", voi.shape)
    return voi, _voi_projections(voi)