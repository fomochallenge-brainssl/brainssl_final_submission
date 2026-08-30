"""fomo_challenge_metrics — callable metric functions for the FOMO26 Challenge."""

from .regression import compute_absolute_error, compute_correlation
from .segmentation import compute_dice_coefficient, compute_normalized_surface_distance
from .classification import (
    compute_auroc,
    compute_ovr_auroc,
    compute_macro_f1,
    compute_ovr_f1,
)
from .fairness import (
    compute_max_disparity,
    compute_fairness_score,
)

__all__ = [
    # classification
    "compute_auroc",
    "compute_ovr_auroc",
    "compute_macro_f1",
    "compute_ovr_f1",
    # regression
    "compute_absolute_error",
    "compute_correlation",
    # segmentation
    "compute_dice_coefficient",
    "compute_normalized_surface_distance",
    # fairness
    "compute_max_disparity",
    "compute_fairness_score",
]
