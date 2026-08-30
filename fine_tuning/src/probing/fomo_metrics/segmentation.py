"""Segmentation metrics using google-deepmind/surface-distance."""

import numpy as np
import surface_distance as surfdist


def compute_dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Dice coefficient between prediction and ground truth masks."""
    try:
        pred_binary = pred_mask.astype(bool)
        gt_binary = gt_mask.astype(bool)

        pred_sum = np.sum(pred_binary)
        gt_sum = np.sum(gt_binary)

        if pred_sum == 0 and gt_sum == 0:
            return 1.0

        if pred_sum == 0 or gt_sum == 0:
            return 0.0

        intersection = np.sum(pred_binary & gt_binary)
        return float(2.0 * intersection / (pred_sum + gt_sum))
    except Exception:
        return 0.0


def compute_normalized_surface_distance(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> float:
    """Compute Normalized Surface Distance (surface Dice at tolerance)."""
    try:
        pred_binary = pred_mask.astype(bool)
        gt_binary = gt_mask.astype(bool)

        if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
            return 0.0

        if np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
            return 0.0

        surface_distances = surfdist.compute_surface_distances(gt_binary, pred_binary, spacing_mm)
        nsd_score = surfdist.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm)

        if np.isnan(nsd_score) or np.isinf(nsd_score):
            return 0.0

        return float(nsd_score)
    except Exception:
        return 0.0



