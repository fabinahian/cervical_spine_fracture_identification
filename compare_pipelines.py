"""
compare_pipelines.py
--------------------
Run the projection-derived pipeline and the 3D-segmentation comparison arm on
the same DICOM study and print them side by side.

Both arms share Stages 1–2 (variance projections, YOLO spine VOI) and Stage 6
(score fusion). They differ in how the vertebrae are delineated:

    projection arm : 2D multi-label segmentation of sagittal / coronal energy
                     projections, extruded and intersected (scripts/stage3–4)
    3D arm         : 3D U-Net (ResNet18d) on the spine VOI, tight per-vertebra
                     crops with non-vertebra voxels set to air (seg3d_pipeline/)

and each arm uses the classifiers trained on its own vertebra VOIs.

Usage
-----
    python compare_pipelines.py --dicom-dir ct_volumes/<study_id>
    python compare_pipelines.py --dicom-dir ct_volumes/<study_id> --arms 3d

Results are saved to ``results/<study_id>_comparison.npy``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from config import BONE_WINDOW, DICOM_DIR, OUT_RESULTS, TARGET_SLICES
from scripts.stage1_projections import generate_projections
from scripts.stage2_voi import detect_voi
from scripts.stage3_segmentation import segment_voi
from scripts.stage4_vertebra_vois import extract_vertebra_vois
from scripts.stage5_classification import classify_vertebrae
from scripts.stage6_ensemble import ensemble
from seg3d_pipeline.config_3d import CLS3D_WEIGHTS
from seg3d_pipeline.stages_3d import extract_vertebra_vois_3d, segment_voi_3d
from utils import apply_window, interpolate_depth, load_dicom_volume

logger = logging.getLogger(__name__)
LABELS = [f"C{i + 1}" for i in range(7)]


def run_arms(dicom_dir: str, arms: list[str]) -> dict:
    study_id = os.path.basename(os.path.normpath(dicom_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def step(name, fn, *a, **k):
        logger.info("[%s]", name)
        t0 = time.perf_counter()
        out = fn(*a, **k)
        logger.info("  Done in %.1fs", time.perf_counter() - t0)
        return out

    volume, _, _ = load_dicom_volume(dicom_dir)
    volume = interpolate_depth(volume, TARGET_SLICES)
    volume = apply_window(volume, BONE_WINDOW[0], BONE_WINDOW[1])
    projections = step("Stage 1 — Variance projections", generate_projections, volume)
    voi, voi_proj = step("Stage 2 — YOLO detection + VOI", detect_voi, projections, volume)
    del volume

    out = {"study_id": study_id}
    if "projection" in arms:
        masks = step("Projection arm — Stage 3 segmentation", segment_voi, voi_proj, device)
        vv = step("Projection arm — Stage 4 vertebra VOIs", extract_vertebra_vois, voi, masks)
        cls = step("Projection arm — Stage 5 classification", classify_vertebrae, vv, device)
        out["projection"] = ensemble(cls)
    if "3d" in arms:
        labelmap = step("3D arm — 3D segmentation", segment_voi_3d, voi, device)
        vv = step("3D arm — vertebra VOIs", extract_vertebra_vois_3d, voi, labelmap)
        cls = step("3D arm — classification", classify_vertebrae, vv, device, CLS3D_WEIGHTS)
        out["3d"] = ensemble(cls)
    return out


def _print(result: dict, arms: list[str]) -> None:
    names = {"projection": "Projection", "3d": "3D seg"}
    print("\n" + "=" * 60)
    print(f"  Study : {result['study_id']}")
    print("-" * 60)
    print("  Vertebra   " + "".join(f"{names[a]:>22s}" for a in arms))
    for lab in LABELS:
        cells = []
        for a in arms:
            v = result[a]["vertebra"][lab]
            cells.append(f"{v['unified_score']:>14.4f}  pred={v['prediction']}")
        print(f"  {lab:9s}" + "".join(f"{c:>22s}" for c in cells))
    print("-" * 60)
    for a in arms:
        p = result[a]["patient"]
        print(f"  {names[a]:11s} patient signal={p['patient_signal']:.4f}  agreement={p['avg_agreement']:.4f}  "
              f"threshold={p['adaptive_threshold']}  FRACTURE={'YES' if p['prediction'] else 'NO'}")
    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare the projection-derived and 3D-segmentation pipelines on one study.")
    ap.add_argument("--dicom-dir", default=DICOM_DIR, metavar="PATH")
    ap.add_argument("--arms", default="projection,3d", help="comma-separated: projection, 3d")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    result = run_arms(args.dicom_dir, arms)
    os.makedirs(OUT_RESULTS, exist_ok=True)
    path = os.path.join(OUT_RESULTS, f"{result['study_id']}_comparison.npy")
    np.save(path, result, allow_pickle=True)
    logger.info("Result saved → %s", path)
    _print(result, arms)


if __name__ == "__main__":
    main()
