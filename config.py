"""
config.py
---------
Central configuration for all paths, model weights, and hyper-parameters.
This is the only file you need to edit before running inference.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

DICOM_DIR: str = r"ct_volumes\1.2.826.0.1.3680043.234"

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

SLICE_CLAHE: bool = True
BONE_WINDOW: tuple[int, int] = (400, 1800)   # (centre, width) in HU
TARGET_SLICES: int = 400

# Slice range trimming the sagittal stack to the cervical region.
SAGITTAL_SLICE_RANGE: tuple[int, int] = (100, 421)

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

YOLO_WEIGHTS: dict[str, str] = {
    "sagittal": r"weights\yolo_sagittal.pt",
    "coronal":  r"weights\yolo_coronal.pt",
    "axial":    r"weights\yolo_axial.pt",
}
VOI_PADDING: int = 20  # voxel margin added around the detected bounding box

# ---------------------------------------------------------------------------
# U-Net segmentation
# ---------------------------------------------------------------------------

SEG_WEIGHTS: dict[str, str] = {
    "sagittal": r"weights\seg_sagittal.pt",
    "coronal":  r"weights\seg_coronal.pt",
}
SEG_INPUT_SIZE:  tuple[int, int] = (256, 256)
SEG_IN_CHANNELS: int   = 1
SEG_NUM_CLASSES: int   = 8      # background + C1–C7
SEG_THRESHOLD:   float = 0.5

# Per-view normalisation statistics (μ, σ) from the training set.
SEG_MEAN: dict[str, list[float]] = {"sagittal": [0.2685], "coronal": [0.2816]}
SEG_STD:  dict[str, list[float]] = {"sagittal": [0.2083], "coronal": [0.2079]}

# ---------------------------------------------------------------------------
# CNN-Transformer classifier
# ---------------------------------------------------------------------------

CLS_WEIGHTS: dict[str, str] = {
    "model1": r"weights\slice_stack_classification.pth",
    "model2": r"weights\proj_stack_classification.pth",
}
CLS_BACKBONE:       str   = "tf_efficientnetv2_s_in21ft1k"
CLS_IN_CHANS:       int   = 5
CLS_OUT_DIM:        int   = 1
CLS_N_SLICE:        int   = 15
CLS_IMAGE_SIZE:     int   = 224
CLS_SLICE_VIEW:     str   = "sagittal"
CLS_PAD:            bool  = True
CLS_DROP_RATE:      float = 0.0
CLS_DROP_RATE_LAST: float = 0.0
CLS_TRANS_LAYERS:   int   = 2
CLS_TRANS_NHEAD:    int   = 8

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUT_RESULTS: str = r"results"