"""Pixel-level evaluation of a fully annotated ROI."""
import numpy as np


def evaluate(labels, target, selected):
    labels = np.asarray(labels)
    target = np.asarray(target, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    if labels.shape != target.shape or labels.shape != selected.shape:
        raise ValueError("Labels, target and selection must have the same shape")
    if not np.isin(labels, [0, 1, 2, 3, 255]).all():
        raise ValueError("Unexpected annotation label")
    if np.any(selected & ~target):
        raise ValueError("Selected pixels must be inside the target")
    if not target.any():
        return {"status": "empty_target"}
    if np.any(target & (labels == 255)):
        return {"status": "reference_incomplete"}
    skin = target & (labels == 1)
    other = target & ((labels == 2) | (labels == 3))
    tp = int((selected & skin).sum())
    fp = int((selected & other).sum())
    fn = int((~selected & skin).sum())
    unknown = int((target & (labels == 0)).sum())
    selected_unknown = int((selected & (labels == 0)).sum())
    selected_count = int(selected.sum())
    reason = None
    if tp + fp == 0:
        reason = "selected_unknown_only" if selected_count else "selected_none"
    return {
        "status": "ok", "tp": tp, "fp": fp, "fn": fn,
        "selected": selected_count, "selected_known": tp + fp,
        "selected_unknown": selected_unknown,
        "unknown_fraction": unknown / int(target.sum()),
        "contamination": fp / (tp + fp) if tp + fp else None,
        "contamination_reason": reason,
        "retention": tp / (tp + fn) if tp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
    }
