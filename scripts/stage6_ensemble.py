# path: scripts/stage6_ensemble.py

"""
Stage 6 — Ensemble fusion.

Vertebra-level : averages both model probabilities and thresholds at 0.5.
Patient-level  : applies an agreement-adaptive threshold to the maximum
                 unified score across all vertebrae.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ensemble"]

NUM_VERTEBRAE   = 7
VERTEBRA_LABELS = [f"C{i + 1}" for i in range(NUM_VERTEBRAE)]


def _vertebra_score_fusion(
    cls_results: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Average per-model probabilities and apply a 0.5 threshold.

    Args:
        cls_results: Output of Stage 5.

    Returns:
        Dictionary mapping vertebra label →
        ``{p1, p2, unified_score, prediction}``.
    """
    names = list(cls_results.keys())
    m1, m2 = cls_results[names[0]], cls_results[names[1]]
    out: dict[str, dict] = {}

    for label in VERTEBRA_LABELS:
        p1 = m1.get(label, float("nan"))
        p2 = m2.get(label, float("nan"))
        if np.isnan(p1) or np.isnan(p2):
            unified, pred = float("nan"), 0
        else:
            unified = (p1 + p2) / 2.0
            pred = int(unified >= 0.5)
        out[label] = {"p1": p1, "p2": p2, "unified_score": unified, "prediction": pred}

    return out


def _patient_adaptive_threshold(vertebra_out: dict[str, dict]) -> dict:
    """Derive a patient-level prediction with an agreement-adaptive threshold.

    Per-vertebra agreement is ``1 − |p1 − p2|``.  The patient signal is the
    maximum unified score.  The threshold is tightened when models agree:

    ================  =========  ==============================
    avg_agreement     threshold  rationale
    ================  =========  ==============================
    > 0.8             0.3        High confidence → sensitive
    > 0.6             0.5        Moderate confidence
    ≤ 0.6             0.7        Low confidence → conservative
    ================  =========  ==============================

    Args:
        vertebra_out: Output of :func:`_vertebra_score_fusion`.

    Returns:
        Dictionary with keys ``patient_signal``, ``avg_agreement``,
        ``adaptive_threshold``, and ``prediction``.
    """
    valid = [
        (v["unified_score"], 1.0 - abs(v["p1"] - v["p2"]))
        for v in vertebra_out.values()
        if not np.isnan(v["unified_score"])
    ]

    if not valid:
        return {"patient_signal": float("nan"), "avg_agreement": float("nan"),
                "adaptive_threshold": float("nan"), "prediction": 0}

    unified_scores, agreement_scores = zip(*valid)
    patient_signal = float(np.max(unified_scores))
    avg_agreement  = float(np.mean(agreement_scores))

    threshold = 0.3 if avg_agreement > 0.8 else (0.5 if avg_agreement > 0.6 else 0.7)

    return {
        "patient_signal":     patient_signal,
        "avg_agreement":      avg_agreement,
        "adaptive_threshold": threshold,
        "prediction":         int(patient_signal > threshold),
    }


def ensemble(cls_results: dict[str, dict[str, float]]) -> dict:
    """Fuse classifier outputs into vertebra- and patient-level predictions.

    Args:
        cls_results: Output of Stage 5.

    Returns:
        Dictionary with keys ``"vertebra"`` and ``"patient"``.
    """
    vertebra_out = _vertebra_score_fusion(cls_results)
    patient_out  = _patient_adaptive_threshold(vertebra_out)

    for label, v in vertebra_out.items():
        s = v["unified_score"]
        logger.info("  %s: score=%s  pred=%d",
                    label, f"{s:.4f}" if not np.isnan(s) else "N/A", v["prediction"])

    logger.info(
        "  Patient: signal=%.4f  agreement=%.4f  threshold=%.1f  pred=%d",
        patient_out["patient_signal"], patient_out["avg_agreement"],
        patient_out["adaptive_threshold"], patient_out["prediction"],
    )
    return {"vertebra": vertebra_out, "patient": patient_out}