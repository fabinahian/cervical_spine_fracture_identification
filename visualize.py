"""
visualize.py
------------
Real-time pipeline visualisation dashboard for the cervical vertebra fracture
detection system. Runs all six inference stages in a background thread and
renders each stage's inputs and outputs live in a dark-themed medical Qt window.

Click any completed sidebar card to revisit its output at any time.

Additional file — does not modify any existing pipeline files.

Usage
-----
    python visualize.py
    python visualize.py --dicom-dir path/to/study
    python visualize.py --verbose
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QSizePolicy, QVBoxLayout, QWidget,
)
from ultralytics import YOLO

from config import BONE_WINDOW, DICOM_DIR, TARGET_SLICES, YOLO_WEIGHTS
from scripts.stage1_projections    import generate_projections
from scripts.stage2_voi            import _build_voi, _projection_to_bgr, _voi_projections, _yolo_bbox
from scripts.stage3_segmentation   import segment_voi
from scripts.stage4_vertebra_vois  import VERTEBRA_LABELS, extract_vertebra_vois
from scripts.stage5_classification import classify_vertebrae
from scripts.stage6_ensemble       import ensemble
from utils import apply_window, interpolate_depth, load_dicom_volume

logger = logging.getLogger(__name__)


# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "accent":  "#00d4aa",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "success": "#3fb950",
    "active":  "#f0883e",
    "pending": "#484f58",
    "danger":  "#f85149",
    "blue":    "#58a6ff",
    "hover":   "#1f2937",
}

STAGE_INFO = [
    ("Variance Projections",    "Sagittal · Coronal · Axial"),
    ("YOLO Detection + VOI",    "Bounding boxes · VOI crop"),
    ("Multilabel Segmentation", "C1–C7 mask overlay"),
    ("Vertebra VOI Extraction", "Per-vertebra 3-D crops"),
    ("Fracture Classification", "CNN-Transformer probabilities"),
    ("Ensemble Fusion",         "Final vertebra + patient result"),
]

# Dark theme for all matplotlib figures.
plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["bg"],
    "text.color":       C["text"],
    "axes.labelcolor":  C["text"],
    "xtick.color":      C["subtext"],
    "ytick.color":      C["subtext"],
    "axes.edgecolor":   C["border"],
    "grid.color":       "#21262d",
    "axes.titlecolor":  C["text"],
    "axes.titlesize":   10,
    "font.family":      "monospace",
})


# ── Figure helpers ────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-8)


def _suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, color=C["accent"], fontsize=13, fontweight="bold", y=0.99)


def _off(ax) -> None:
    ax.axis("off")


def _draw_bbox(
    ax,
    bbox: tuple[int, int, int, int],
    label: str,
    color: str = C["accent"],
) -> None:
    y_min, y_max, x_min, x_max = bbox
    rect = mpatches.FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=2", linewidth=2,
        edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        x_min + 4, max(y_min - 8, 0), label,
        color=color, fontsize=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=1", fc=C["panel"], ec=color, alpha=0.85),
    )


# ── Stage figure factories ────────────────────────────────────────────────────

def _fig_loading(volume: np.ndarray) -> Figure:
    """Three orthogonal mid-slices of the raw CT volume."""
    mid = [s // 2 for s in volume.shape]
    panels = {
        "Axial":    volume[mid[0], :, :],
        "Coronal":  volume[:, mid[1], :],
        "Sagittal": volume[:, :, mid[2]],
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    _suptitle(fig, "CT Volume — Orthogonal Preview")
    for ax, (title, sl) in zip(axes, panels.items()):
        ax.imshow(_norm(sl), cmap="bone", aspect="auto")
        ax.set_title(title, color=C["subtext"])
        _off(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_stage1(projections: dict) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    _suptitle(fig, "Stage 1 — Variance Projections")
    for ax, view in zip(axes, ("sagittal", "coronal", "axial")):
        ax.imshow(_norm(projections[view]), cmap="bone", aspect="auto")
        ax.set_title(view.capitalize(), color=C["subtext"])
        _off(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_stage2(
    projections: dict,
    bboxes: dict,
    voi_proj: dict,
) -> Figure:
    fig = plt.figure(figsize=(14, 9))
    _suptitle(fig, "Stage 2 — YOLO Detection & Volume of Interest")
    bbox_colors = {"sagittal": C["accent"], "coronal": "#ff9f43", "axial": "#ee5a24"}

    for col, view in enumerate(("sagittal", "coronal", "axial")):
        ax = fig.add_subplot(2, 3, col + 1)
        ax.imshow(_norm(projections[view]), cmap="bone", aspect="auto")
        if view in bboxes:
            _draw_bbox(ax, bboxes[view], view.capitalize(), bbox_colors[view])
        ax.set_title(f"{view.capitalize()}  —  Detection", color=C["subtext"])
        _off(ax)

    for col, view in enumerate(("sagittal", "coronal")):
        ax = fig.add_subplot(2, 3, col + 4)
        if view in voi_proj:
            ax.imshow(_norm(voi_proj[view]), cmap="inferno", aspect="auto")
            ax.set_title(f"VOI — {view.capitalize()} Energy Projection", color=C["subtext"])
        else:
            ax.set_title(f"VOI — {view.capitalize()}  (pending…)", color=C["pending"])
        _off(ax)

    ax_info = fig.add_subplot(2, 3, 6)
    _off(ax_info)
    lines = ["Detection Summary", "─" * 30]
    for view, bb in bboxes.items():
        h, w = bb[1] - bb[0], bb[3] - bb[2]
        lines.append(f"  {view:>9}  h={h:>4}  w={w:>4}")
    ax_info.text(
        0.05, 0.92, "\n".join(lines),
        transform=ax_info.transAxes, va="top", color=C["text"],
        fontsize=8, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc=C["panel"], ec=C["border"], alpha=0.9),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_stage3(voi_proj: dict, seg_masks: dict) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _suptitle(fig, "Stage 3 — Multilabel Segmentation  (C1–C7 overlay)")
    cmap = plt.get_cmap("tab10", 8)

    for ax, view in zip(axes, ("sagittal", "coronal")):
        proj = _norm(voi_proj[view])
        mask = seg_masks[view]
        n_cls = mask.shape[2]
        rgb = np.zeros((*mask.shape[:2], 3), dtype=np.float64)
        for i in range(n_cls):
            rgb += mask[:, :, i][..., None] * np.array(cmap(i)[:3])
        rgb = np.clip(rgb, 0, 1)
        ax.imshow(proj, cmap="bone", aspect="auto")
        ax.imshow(rgb, alpha=0.55, aspect="auto")
        ax.set_title(view.capitalize(), color=C["subtext"])
        _off(ax)

    handles = [
        mpatches.Patch(color=cmap(i)[:3], label="Background" if i == 0 else f"C{i}")
        for i in range(8)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               framealpha=0.15, labelcolor=C["text"], fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    return fig


def _vertebra_axial_proj(voi: np.ndarray) -> np.ndarray:
    """Axial variance projection of a vertebra VOI.

    VOI shape is (H, D, W). Projecting along H (axis 0) collapses superior–
    inferior, yielding a (D, W) image — the axial bird's-eye view.
    """
    return _norm(np.var(voi.astype(np.float64), axis=0))


def _fig_stage4(vertebra_vois: dict) -> Figure:
    n    = len(vertebra_vois)
    cols = min(n, 4)
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5 + 1))
    _suptitle(fig, "Stage 4 — Per-Vertebra VOI  (Axial Variance Projection)")
    axes_flat = np.array(axes).flatten()

    for i, (label, voi) in enumerate(vertebra_vois.items()):
        ax = axes_flat[i]
        proj = _vertebra_axial_proj(voi)
        ax.imshow(proj, cmap="bone", aspect="auto")
        ax.set_title(label, color=C["accent"], fontsize=12, fontweight="bold")
        ax.text(0.03, 0.03, f"VOI {voi.shape}", transform=ax.transAxes,
                color=C["subtext"], fontsize=7)
        _off(ax)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_stage5(cls_results: dict) -> Figure:
    names   = list(cls_results.keys())
    p1      = [cls_results[names[0]].get(l, float("nan")) for l in VERTEBRA_LABELS]
    p2      = [cls_results[names[1]].get(l, float("nan")) for l in VERTEBRA_LABELS]
    unified = [
        (a + b) / 2.0 if not (np.isnan(a) or np.isnan(b)) else float("nan")
        for a, b in zip(p1, p2)
    ]
    x, w = np.arange(len(VERTEBRA_LABELS)), 0.25
    u_colors = [C["danger"] if not np.isnan(u) and u >= 0.5 else C["success"]
                for u in unified]

    fig, ax = plt.subplots(figsize=(14, 6))
    _suptitle(fig, "Stage 5 — Fracture Classification Probabilities")
    ax.bar(x - w, p1,      w, label=f"{names[0]}", color=C["subtext"], alpha=0.75)
    ax.bar(x,     p2,      w, label=f"{names[1]}", color=C["blue"],    alpha=0.75)
    ax.bar(x + w, unified, w, label="Unified",     color=u_colors,     alpha=0.95)
    ax.axhline(0.5, color=C["active"], linestyle="--", lw=1.5, label="Threshold 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels(VERTEBRA_LABELS, color=C["text"])
    ax.set_ylabel("Fracture Probability")
    ax.set_ylim(0, 1.1)
    ax.legend(framealpha=0.2, labelcolor=C["text"])
    ax.grid(axis="y", alpha=0.25)
    for xi, u in zip(x + w, unified):
        if not np.isnan(u):
            ax.text(xi, u + 0.02, f"{u:.2f}", ha="center", va="bottom",
                    fontsize=7, color=C["text"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _fig_stage6(result: dict) -> Figure:
    vertebra = result["vertebra"]
    patient  = result["patient"]

    fig = plt.figure(figsize=(14, 6))
    _suptitle(fig, "Stage 6 — Ensemble Fusion Results")

    ax_t = fig.add_subplot(1, 2, 1)
    _off(ax_t)
    col_labels = ["Vertebra", "Model 1", "Model 2", "Unified", "Prediction"]
    rows_data, cell_clr = [], []
    for lbl in VERTEBRA_LABELS:
        v = vertebra[lbl]
        p1 = f"{v['p1']:.3f}" if not np.isnan(v['p1']) else "N/A"
        p2 = f"{v['p2']:.3f}" if not np.isnan(v['p2']) else "N/A"
        us = f"{v['unified_score']:.3f}" if not np.isnan(v['unified_score']) else "N/A"
        pred = "FRACTURE" if v["prediction"] else "Normal"
        rows_data.append([lbl, p1, p2, us, pred])
        cell_clr.append(["#3d0000" if v["prediction"] else "#0d2818"] * 5)

    tbl = ax_t.table(cellText=rows_data, colLabels=col_labels,
                     cellColours=cell_clr, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["border"])
        cell.set_text_props(color=C["text"])
        if row == 0:
            cell.set_facecolor(C["panel"])
            cell.set_text_props(color=C["accent"], fontweight="bold")
    ax_t.set_title("Vertebra-level Results", color=C["subtext"], pad=12)

    ax_p = fig.add_subplot(1, 2, 2)
    _off(ax_p)
    pred_color = C["danger"] if patient["prediction"] else C["success"]
    pred_label = "FRACTURE DETECTED" if patient["prediction"] else "NO FRACTURE"
    ax_p.text(
        0.5, 0.70, pred_label,
        transform=ax_p.transAxes, ha="center", va="center",
        fontsize=22, fontweight="bold", color=pred_color,
        bbox=dict(boxstyle="round,pad=0.7", fc=C["panel"],
                  ec=pred_color, linewidth=3, alpha=0.92),
    )
    ax_p.text(
        0.5, 0.28,
        f"Patient signal     {patient['patient_signal']:.4f}\n"
        f"Avg agreement      {patient['avg_agreement']:.4f}\n"
        f"Adaptive threshold {patient['adaptive_threshold']}",
        transform=ax_p.transAxes, ha="center", va="center",
        fontsize=11, color=C["text"], fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc=C["panel"], ec=C["border"], alpha=0.85),
    )
    ax_p.set_title("Patient-level Result", color=C["subtext"], pad=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# Figure builder dispatch — keyed by stage index (-1 = volume preview).
_FIG_BUILDERS = {
    -1: lambda d: _fig_loading(d["volume"]),
     0: lambda d: _fig_stage1(d["projections"]),
     1: lambda d: _fig_stage2(d["projections"], d["bboxes"], d["voi_proj"]),
     2: lambda d: _fig_stage3(d["voi_proj"], d["seg_masks"]),
     3: lambda d: _fig_stage4(d["vertebra_vois"]),
     4: lambda d: _fig_stage5(d["cls_results"]),
     5: lambda d: _fig_stage6(d["result"]),
}


# ── Pipeline worker ───────────────────────────────────────────────────────────
class PipelineWorker(QThread):
    stage_started  = pyqtSignal(int)
    stage_done     = pyqtSignal(int, float)
    stage_data     = pyqtSignal(int, object)
    bbox_detected  = pyqtSignal(str, object)   # (view, bbox) — fires per-view
    error          = pyqtSignal(str)

    def __init__(self, dicom_dir: str, device) -> None:
        super().__init__()
        self.dicom_dir = dicom_dir
        self.device    = device

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            logger.exception("Pipeline error")
            self.error.emit(str(exc))

    def _run(self) -> None:
        volume, _, _ = load_dicom_volume(self.dicom_dir)
        volume = apply_window(volume, BONE_WINDOW[0], BONE_WINDOW[1])
        volume = interpolate_depth(volume, TARGET_SLICES)
        self.stage_data.emit(-1, {"volume": volume})

        self.stage_started.emit(0)
        t0 = time.perf_counter()
        projections = generate_projections(volume)
        self.stage_done.emit(0, time.perf_counter() - t0)
        self.stage_data.emit(0, {"projections": projections})

        self.stage_started.emit(1)
        t0 = time.perf_counter()
        yolo_models = {view: YOLO(w) for view, w in YOLO_WEIGHTS.items()}
        bboxes: dict[str, tuple] = {}
        for view in ("sagittal", "coronal", "axial"):
            bboxes[view] = _yolo_bbox(_projection_to_bgr(projections[view]), yolo_models[view])
            # Emit immediately after each detection so the UI can draw it live.
            self.bbox_detected.emit(view, dict(bboxes))
        voi      = _build_voi(bboxes, volume)
        voi_proj = _voi_projections(voi)
        self.stage_done.emit(1, time.perf_counter() - t0)
        self.stage_data.emit(1, {"projections": projections, "bboxes": bboxes,
                                  "voi": voi, "voi_proj": voi_proj})

        self.stage_started.emit(2)
        t0 = time.perf_counter()
        seg_masks = segment_voi(voi_proj, self.device)
        self.stage_done.emit(2, time.perf_counter() - t0)
        self.stage_data.emit(2, {"voi_proj": voi_proj, "seg_masks": seg_masks})

        self.stage_started.emit(3)
        t0 = time.perf_counter()
        vertebra_vois = extract_vertebra_vois(voi, seg_masks)
        self.stage_done.emit(3, time.perf_counter() - t0)
        self.stage_data.emit(3, {"vertebra_vois": vertebra_vois})

        self.stage_started.emit(4)
        t0 = time.perf_counter()
        cls_results = classify_vertebrae(vertebra_vois, self.device)
        self.stage_done.emit(4, time.perf_counter() - t0)
        self.stage_data.emit(4, {"cls_results": cls_results})

        self.stage_started.emit(5)
        t0 = time.perf_counter()
        result = ensemble(cls_results)
        result["study_id"] = os.path.basename(os.path.normpath(self.dicom_dir))
        self.stage_done.emit(5, time.perf_counter() - t0)
        self.stage_data.emit(5, {"result": result})


# ── Sidebar stage card ────────────────────────────────────────────────────────
class StageCard(QWidget):
    """Clickable sidebar card. Emits ``clicked`` only when stage is complete."""
    clicked = pyqtSignal(int)

    def __init__(self, idx: int, title: str, subtitle: str) -> None:
        super().__init__()
        self._idx      = idx
        self._done     = False
        self._selected = False
        self.setFixedHeight(62)
        self.setCursor(Qt.ArrowCursor)
        self._apply_style(C["border"])

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setFixedWidth(18)
        self._dot.setFont(QFont("monospace", 13))
        self._dot.setStyleSheet(f"color: {C['pending']};")

        text = QVBoxLayout()
        text.setSpacing(2)
        self._title = QLabel(f"Stage {idx + 1} — {title}")
        self._title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self._title.setStyleSheet(f"color: {C['subtext']};")
        self._sub = QLabel(subtitle)
        self._sub.setFont(QFont("Segoe UI", 7))
        self._sub.setStyleSheet(f"color: {C['pending']};")
        text.addWidget(self._title)
        text.addWidget(self._sub)

        self._time = QLabel("")
        self._time.setFont(QFont("monospace", 7))
        self._time.setStyleSheet(f"color: {C['subtext']};")
        self._time.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(self._dot)
        lay.addLayout(text)
        lay.addStretch()
        lay.addWidget(self._time)

    def _apply_style(self, border_color: str, bg: Optional[str] = None) -> None:
        bg = bg or C["panel"]
        self.setStyleSheet(
            f"background: {bg}; border-radius: 8px;"
            f"border: 1px solid {border_color};"
        )

    def set_active(self) -> None:
        self._dot.setStyleSheet(f"color: {C['active']};")
        self._title.setStyleSheet(f"color: {C['text']};")
        self._sub.setStyleSheet(f"color: {C['active']};")
        self._apply_style(C["active"])

    def set_done(self, elapsed: float) -> None:
        self._done = True
        self.setCursor(Qt.PointingHandCursor)
        self._dot.setStyleSheet(f"color: {C['success']};")
        self._title.setStyleSheet(f"color: {C['text']};")
        self._sub.setStyleSheet(f"color: {C['subtext']};")
        self._time.setText(f"{elapsed:.1f}s")
        self._apply_style(C["success"])

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if self._done:
            border = C["accent"] if selected else C["success"]
            bg     = C["hover"] if selected else C["panel"]
            self._apply_style(border, bg)

    def mousePressEvent(self, event) -> None:
        if self._done and event.button() == Qt.LeftButton:
            self.clicked.emit(self._idx)

    def enterEvent(self, event) -> None:
        if self._done and not self._selected:
            self._apply_style(C["accent"], C["hover"])

    def leaveEvent(self, event) -> None:
        if self._done and not self._selected:
            self._apply_style(C["success"])


# ── Main window ───────────────────────────────────────────────────────────────
class VisualizerWindow(QMainWindow):
    def __init__(self, dicom_dir: str) -> None:
        super().__init__()
        import torch

        self.dicom_dir     = dicom_dir
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_start       = time.perf_counter()
        self._canvas: Optional[FigureCanvas] = None
        self._placeholder: Optional[QLabel]  = None
        self._selected_idx: Optional[int]    = None
        # Cache: stage_idx → raw data payload for on-demand re-render.
        self._data_cache: dict[int, dict]    = {}

        self.setWindowTitle("🔬  Cervical Vertebra Fracture Detection — Pipeline Visualiser")
        self.setMinimumSize(1340, 820)
        self._apply_palette()
        self._build_ui()
        self._start_pipeline()

    def _apply_palette(self) -> None:
        pal = QPalette()
        pal.setColor(QPalette.Window,     QColor(C["bg"]))
        pal.setColor(QPalette.WindowText, QColor(C["text"]))
        pal.setColor(QPalette.Base,       QColor(C["panel"]))
        pal.setColor(QPalette.Text,       QColor(C["text"]))
        self.setPalette(pal)
        self.setStyleSheet(f"background: {C['bg']};")

    def _build_ui(self) -> None:
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QHBoxLayout(root_widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(265)
        sidebar.setStyleSheet(f"background: {C['bg']};")
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(6)

        header = QLabel("PIPELINE STAGES")
        header.setFont(QFont("Segoe UI", 8, QFont.Bold))
        header.setStyleSheet(f"color: {C['accent']}; letter-spacing: 2px;")
        sb.addWidget(header)

        hint = QLabel("Click a completed stage to revisit it")
        hint.setFont(QFont("Segoe UI", 7))
        hint.setStyleSheet(f"color: {C['pending']};")
        sb.addWidget(hint)
        sb.addSpacing(4)

        self._cards: list[StageCard] = []
        for i, (title, sub) in enumerate(STAGE_INFO):
            card = StageCard(i, title, sub)
            card.clicked.connect(self._on_card_clicked)
            self._cards.append(card)
            sb.addWidget(card)

        sb.addStretch()

        self._status = QLabel("Loading study…")
        self._status.setFont(QFont("monospace", 8))
        self._status.setStyleSheet(f"color: {C['subtext']};")
        self._status.setWordWrap(True)
        sb.addWidget(self._status)
        sb.addWidget(self._make_sep())

        dev_lbl = QLabel(f"Device: {self.device}")
        dev_lbl.setFont(QFont("monospace", 7))
        dev_lbl.setStyleSheet(f"color: {C['pending']};")
        sb.addWidget(dev_lbl)

        root.addWidget(sidebar)

        # ── Canvas frame ──────────────────────────────────────────────────
        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(
            f"background: {C['panel']}; border-radius: 10px;"
            f"border: 1px solid {C['border']};"
        )
        self._canvas_lay = QVBoxLayout(canvas_frame)
        self._canvas_lay.setContentsMargins(4, 4, 4, 4)

        self._placeholder = QLabel("Initialising pipeline…")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setFont(QFont("Segoe UI", 14))
        self._placeholder.setStyleSheet(f"color: {C['subtext']};")
        self._canvas_lay.addWidget(self._placeholder)

        root.addWidget(canvas_frame, 1)

        self.statusBar().setStyleSheet(
            f"background: {C['panel']}; color: {C['subtext']}; font-family: monospace;"
        )
        self.statusBar().showMessage(f"Study: {os.path.basename(self.dicom_dir)}")

    @staticmethod
    def _make_sep() -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {C['border']};")
        return sep

    def _show_figure(self, fig: Figure) -> None:
        if self._canvas is not None:
            self._canvas_lay.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        if self._placeholder is not None:
            self._canvas_lay.removeWidget(self._placeholder)
            self._placeholder.deleteLater()
            self._placeholder = None

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_lay.addWidget(canvas)
        self._canvas = canvas
        canvas.draw()
        plt.close(fig)

    def _render_stage(self, idx: int) -> None:
        """Build and display the figure for *idx* from the cached payload."""
        if idx not in _FIG_BUILDERS or idx not in self._data_cache:
            return
        self._show_figure(_FIG_BUILDERS[idx](self._data_cache[idx]))

    def _set_selected_card(self, idx: int) -> None:
        """Highlight *idx* card and deselect all others."""
        if self._selected_idx is not None:
            self._cards[self._selected_idx].set_selected(False)
        self._selected_idx = idx
        self._cards[idx].set_selected(True)

    def _start_pipeline(self) -> None:
        self._worker = PipelineWorker(self.dicom_dir, self.device)
        self._worker.stage_started.connect(self._on_stage_started)
        self._worker.stage_done.connect(self._on_stage_done)
        self._worker.stage_data.connect(self._on_stage_data)
        self._worker.bbox_detected.connect(self._on_bbox_detected)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Qt slots ──────────────────────────────────────────────────────────
    def _on_card_clicked(self, idx: int) -> None:
        self._set_selected_card(idx)
        self._render_stage(idx)
        self.statusBar().showMessage(
            f"Study: {os.path.basename(self.dicom_dir)}  |  "
            f"Viewing Stage {idx + 1} — {STAGE_INFO[idx][0]}"
        )

    def _on_stage_started(self, idx: int) -> None:
        self._cards[idx].set_active()
        self._status.setText(f"Running Stage {idx + 1} — {STAGE_INFO[idx][0]}…")
        elapsed = time.perf_counter() - self.t_start
        self.statusBar().showMessage(
            f"Study: {os.path.basename(self.dicom_dir)}  |  "
            f"Stage {idx + 1}/{len(STAGE_INFO)}  |  Elapsed: {elapsed:.1f}s"
        )

    def _on_stage_done(self, idx: int, elapsed: float) -> None:
        self._cards[idx].set_done(elapsed)

    def _on_stage_data(self, idx: int, data: dict) -> None:
        # Cache the raw data so the card click can re-render any time.
        self._data_cache[idx] = data

        # Only auto-display if the user hasn't navigated away to a past stage.
        if self._selected_idx is None or self._selected_idx == idx:
            self._set_selected_card(max(idx, 0))
            self._render_stage(idx)

        if idx == 5:
            total = time.perf_counter() - self.t_start
            self._status.setText(f"Pipeline complete  ({total:.1f}s total)")
            self.statusBar().showMessage(
                f"Study: {os.path.basename(self.dicom_dir)}  |  "
                f"Complete  |  Total: {total:.1f}s"
            )

    def _on_bbox_detected(self, view: str, bboxes_so_far: dict) -> None:
        """Re-render Stage 2 each time a new YOLO bbox arrives.

        ``bboxes_so_far`` contains only the views detected so far, so boxes
        appear one by one as each model finishes.  The VOI projections row is
        left empty until Stage 2 fully completes and emits ``stage_data``.
        """
        projections = self._data_cache.get(0, {}).get("projections", {})
        if not projections:
            return
        # Pass an empty voi_proj so the energy-projection row stays blank.
        fig = _fig_stage2(projections, bboxes_so_far, voi_proj={})
        self._show_figure(fig)
        self.statusBar().showMessage(
            f"Study: {os.path.basename(self.dicom_dir)}  |  "
            f"Stage 2 — YOLO detected: {', '.join(bboxes_so_far.keys())}"
        )

    def _on_error(self, msg: str) -> None:
        self._status.setText(f"Error: {msg}")
        self._status.setStyleSheet(f"color: {C['danger']};")
        self.statusBar().showMessage(f"Pipeline failed: {msg}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time pipeline visualisation dashboard.",
    )
    parser.add_argument("--dicom-dir", default=DICOM_DIR, metavar="PATH")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = VisualizerWindow(args.dicom_dir)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()