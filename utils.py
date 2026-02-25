"""
utils.py
--------
Shared DICOM I/O, volumetric pre-processing, and image utility functions.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pydicom
from scipy.ndimage import zoom
from skimage import exposure

logger = logging.getLogger(__name__)

__all__ = [
    "load_dicom_volume",
    "apply_window",
    "interpolate_depth",
    "normalize_slice",
    "clahe_volume",
    "pad_numpy_to_square",
]


def load_dicom_volume(
    folder: str,
) -> tuple[np.ndarray, Optional[float], Optional[float]]:
    """Load a DICOM study folder into a 3-D Hounsfield-Unit volume.

    Slices are sorted in descending ``ImagePositionPatient[2]`` order
    (superior → inferior) and rescaled using per-slice DICOM tags.

    Args:
        folder: Path to a directory containing .dcm files for one study.
            Non-DICOM files are silently skipped.

    Returns:
        ``(volume, window_center, window_width)`` where ``volume`` is a
        float64 array of shape ``(D, H, W)`` in HU, and the window values
        come from the first slice that exposes them (or ``None``).

    Raises:
        ValueError: If no readable DICOM files are found in *folder*.
    """
    slices: list[pydicom.Dataset] = []
    window_center: Optional[float] = None
    window_width:  Optional[float] = None

    for fname in sorted(os.listdir(folder)):
        try:
            ds = pydicom.dcmread(os.path.join(folder, fname))
            slices.append(ds)
            if window_center is None and "WindowCenter" in ds:
                wc = ds.WindowCenter
                window_center = float(wc if np.isscalar(wc) else wc[0])
            if window_width is None and "WindowWidth" in ds:
                ww = ds.WindowWidth
                window_width = float(ww if np.isscalar(ww) else ww[0])
        except Exception:
            logger.debug("Skipping non-DICOM file: %s", fname)

    if not slices:
        raise ValueError(f"No readable DICOM files found in '{folder}'.")

    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]), reverse=True)
    volume = np.stack(
        [s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept)
         for s in slices],
        axis=0,
    )
    return volume, window_center, window_width


def apply_window(
    volume: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Clip *volume* to a radiological display window.

    Args:
        volume: Input array in Hounsfield Units, any shape.
        window_center: Centre of the display window (HU).
        window_width: Full width of the display window (HU).

    Returns:
        Clipped array with the same shape and dtype as *volume*.
    """
    return np.clip(volume, window_center - window_width / 2.0,
                           window_center + window_width / 2.0)


def interpolate_depth(
    volume: np.ndarray,
    target_slices: int = 400,
    order: int = 3,
) -> np.ndarray:
    """Up-sample *volume* along the depth axis to *target_slices*.

    Returns the input unchanged if ``volume.shape[0] >= target_slices``.

    Args:
        volume: 3-D array of shape ``(D, H, W)``.
        target_slices: Desired depth after interpolation.
        order: Spline interpolation order (default: cubic).

    Returns:
        Array of shape ``(target_slices, H, W)``.
    """
    if volume.shape[0] >= target_slices:
        return volume
    return zoom(volume, (target_slices / volume.shape[0], 1, 1), order=order)


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2-D slice to ``[0, 1]``.

    Returns a zero array for constant slices to avoid division by zero.

    Args:
        sl: 2-D array of any numeric dtype.

    Returns:
        float64 array in ``[0, 1]`` with the same shape.
    """
    mn, mx = float(sl.min()), float(sl.max())
    if mx - mn < 1e-8:
        return np.zeros_like(sl, dtype=np.float64)
    return (sl.astype(np.float64) - mn) / (mx - mn)


def clahe_volume(volume: np.ndarray) -> np.ndarray:
    """Apply CLAHE independently to every slice of a 3-D volume.

    Args:
        volume: 3-D array of shape ``(D, H, W)`` with values in ``[0, 1]``.

    Returns:
        CLAHE-enhanced volume with the same shape.
    """
    out = volume.copy()
    for i in range(out.shape[0]):
        out[i] = exposure.equalize_adapthist(normalize_slice(out[i]))
    return out


def pad_numpy_to_square(image: np.ndarray) -> np.ndarray:
    """Zero-pad a ``(H, W, C)`` array to ``(max(H,W), max(H,W), C)``.

    Args:
        image: Array of shape ``(H, W, C)``.

    Returns:
        Square array with the input centred and zero-padded.
    """
    h, w, c = image.shape
    if h == w:
        return image
    size = max(h, w)
    padded = np.zeros((size, size, c), dtype=image.dtype)
    ph, pw = (size - h) // 2, (size - w) // 2
    padded[ph:ph + h, pw:pw + w, :] = image
    return padded