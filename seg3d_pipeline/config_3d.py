"""
seg3d_pipeline/config_3d.py
---------------------------
Configuration of the comparison arm that replaces the projection-based
vertebra segmentation (Stages 3–4) with a 3D U-Net (ResNet18d encoder, the
architecture of the RSNA 2022 winning solution) trained on the same 87
voxel-annotated patients. Stages 1–2 (spine VOI) and 6 (fusion) are shared
with the main pipeline; Stage 5 uses classifiers trained on 3D-mask VOIs.
"""

from __future__ import annotations

# 3D U-Net: the five cross-validation fold models are averaged (probabilities).
SEG3D_WEIGHTS: list[str] = [rf"weights_3d\seg3d_resnet18d_fold{k}.pth" for k in range(5)]
SEG3D_BACKBONE:   str = "resnet18d"
SEG3D_N_BLOCKS:   int = 4
SEG3D_OUT_DIM:    int = 7                 # C1–C7
SEG3D_INPUT_SIZE: tuple[int, int, int] = (128, 128, 128)
SEG3D_THRESHOLD:  float = 0.5

# Voxels of the crop that do not belong to the target vertebra are set to air.
BACKGROUND_HU: float = -1000.0

# Classifiers trained on 3D-mask VOIs (fold-2 models, same architecture and
# preprocessing as the main pipeline's classifiers).
CLS3D_WEIGHTS: dict[str, str] = {
    "model1": r"weights_3d\slice_stack_classification_3dseg.pth",
    "model2": r"weights_3d\proj_stack_classification_3dseg.pth",
}
