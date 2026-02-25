"""
run.py
------
Command-line entry point for the cervical vertebra fracture detection pipeline.

Usage
-----
    # Use the study path set in config.py
    python run.py

    # Override the study directory at runtime
    python run.py --dicom-dir path/to/study

    # Save all intermediate outputs (projections, masks, VOIs, etc.)
    python run.py --save-intermediates

    # Enable verbose (DEBUG) logging
    python run.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import BONE_WINDOW, DICOM_DIR, OUT_RESULTS, TARGET_SLICES
from scripts.stage1_projections   import generate_projections
from scripts.stage2_voi           import detect_voi
from scripts.stage3_segmentation  import segment_voi
from scripts.stage4_vertebra_vois import extract_vertebra_vois
from scripts.stage5_classification import classify_vertebrae
from scripts.stage6_ensemble      import ensemble
from utils import apply_window, interpolate_depth, load_dicom_volume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate savers
# ---------------------------------------------------------------------------

def _save_intermediates(
    study_id: str,
    base_dir: str,
    projections: dict,
    voi: np.ndarray,
    voi_projections: dict,
    seg_masks: dict,
    vertebra_vois: dict,
    cls_results: dict,
) -> None:
    """Save all intermediate pipeline outputs to *base_dir/study_id/*.

    Directory layout::

        <base_dir>/
        └── <study_id>/
            ├── projections/          # Stage 1 — variance projections
            │   ├── sagittal.png
            │   ├── coronal.png
            │   └── axial.png
            ├── voi/                  # Stage 2 — VOI volume + projections
            │   ├── volume.npy
            │   ├── sagittal.png
            │   └── coronal.png
            ├── seg_masks/            # Stage 3 — segmentation masks
            │   ├── sagittal.npy
            │   └── coronal.npy
            ├── vertebra_vois/        # Stage 4 — per-vertebra crops
            │   ├── C1.npy
            │   └── ...
            └── cls_results.npy       # Stage 5 — raw classifier outputs

    Args:
        study_id: Basename of the DICOM folder, used as the sub-directory name.
        base_dir: Root output directory (``intermediates/`` by default).
        projections: Stage 1 output.
        voi: Stage 2 VOI volume.
        voi_projections: Stage 2 VOI projections.
        seg_masks: Stage 3 output.
        vertebra_vois: Stage 4 output.
        cls_results: Stage 5 output.
    """
    root = os.path.join(base_dir, study_id)

    # Stage 1 — variance projections
    proj_dir = os.path.join(root, "projections")
    os.makedirs(proj_dir, exist_ok=True)
    for view, proj in projections.items():
        plt.imsave(os.path.join(proj_dir, f"{view}.png"), proj, cmap="gray")
    logger.info("  Saved Stage 1 projections → %s", proj_dir)

    # Stage 2 — VOI volume + energy projections
    voi_dir = os.path.join(root, "voi")
    os.makedirs(voi_dir, exist_ok=True)
    np.save(os.path.join(voi_dir, "volume.npy"), voi)
    for view, proj in voi_projections.items():
        plt.imsave(os.path.join(voi_dir, f"{view}.png"), proj, cmap="gray")
    logger.info("  Saved Stage 2 VOI → %s", voi_dir)

    # Stage 3 — segmentation masks
    seg_dir = os.path.join(root, "seg_masks")
    os.makedirs(seg_dir, exist_ok=True)
    for view, mask in seg_masks.items():
        np.save(os.path.join(seg_dir, f"{view}.npy"), mask)
    logger.info("  Saved Stage 3 seg masks → %s", seg_dir)

    # Stage 4 — per-vertebra VOI crops
    vert_dir = os.path.join(root, "vertebra_vois")
    os.makedirs(vert_dir, exist_ok=True)
    for label, vol in vertebra_vois.items():
        np.save(os.path.join(vert_dir, f"{label}.npy"), vol)
    logger.info("  Saved Stage 4 vertebra VOIs → %s", vert_dir)

    # Stage 5 — raw classifier probabilities
    np.save(os.path.join(root, "cls_results.npy"), cls_results)
    logger.info("  Saved Stage 5 cls results → %s", root)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(dicom_dir: str, save_intermediates: bool = False) -> dict:
    """Run the full six-stage inference pipeline on one DICOM study.

    Args:
        dicom_dir: Path to a DICOM study folder.
        save_intermediates: If ``True``, save all intermediate outputs to
            ``intermediates/<study_id>/``.

    Returns:
        Result dictionary with keys ``"study_id"``, ``"vertebra"``,
        and ``"patient"``.
    """
    study_id = os.path.basename(os.path.normpath(dicom_dir))
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start  = time.perf_counter()
    logger.info("Study: %s  |  Device: %s", study_id, device)

    def step(name: str, fn, *args, **kwargs):
        logger.info("[%s]", name)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        logger.info("  Done in %.1fs", time.perf_counter() - t0)
        return out

    volume, _, _ = load_dicom_volume(dicom_dir)
    volume = apply_window(volume, BONE_WINDOW[0], BONE_WINDOW[1])
    volume = interpolate_depth(volume, TARGET_SLICES)

    projections         = step("Stage 1 — Variance projections",    generate_projections, volume)
    voi, voi_proj       = step("Stage 2 — YOLO detection + VOI",    detect_voi, projections, volume)
    seg_masks           = step("Stage 3 — Segmentation",            segment_voi, voi_proj, device)
    vertebra_vois       = step("Stage 4 — Vertebra VOI extraction", extract_vertebra_vois, voi, seg_masks)
    cls_results         = step("Stage 5 — Classification",          classify_vertebrae, vertebra_vois, device)
    result              = step("Stage 6 — Ensemble",                ensemble, cls_results)
    result["study_id"]  = study_id

    logger.info("Total time: %.1fs", time.perf_counter() - t_start)

    # Final result
    os.makedirs(OUT_RESULTS, exist_ok=True)
    out_path = os.path.join(OUT_RESULTS, f"{study_id}_result.npy")
    np.save(out_path, result)
    logger.info("Result saved → %s", out_path)

    # Intermediates (optional)
    if save_intermediates:
        logger.info("Saving intermediate outputs...")
        _save_intermediates(
            study_id, "intermediates",
            projections, voi, voi_proj,
            seg_masks, vertebra_vois, cls_results,
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(result: dict) -> None:
    print("\n" + "=" * 52)
    print(f"  Study : {result['study_id']}")
    print("-" * 52)
    print("  Vertebra-level results:")
    for label, v in result["vertebra"].items():
        s = v["unified_score"]
        print(f"    {label}  score={'N/A   ' if np.isnan(s) else f'{s:.4f}'}  "
              f"pred={v['prediction']}")
    print("-" * 52)
    p = result["patient"]
    print(f"  Patient  signal={p['patient_signal']:.4f}  "
          f"agreement={p['avg_agreement']:.4f}  "
          f"threshold={p['adaptive_threshold']}  "
          f"FRACTURE={'YES' if p['prediction'] else 'NO'}")
    print("=" * 52 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cervical vertebra fracture detection — single-study inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dicom-dir", default=DICOM_DIR, metavar="PATH",
        help="Path to a DICOM study folder.",
    )
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Save projections, VOI, masks, and vertebra crops to intermediates/<study_id>/.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_pipeline(args.dicom_dir, save_intermediates=args.save_intermediates)
    _print_summary(result)


if __name__ == "__main__":
    main(sys.argv[1:])