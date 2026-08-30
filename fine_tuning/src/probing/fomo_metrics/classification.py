"""Classification metrics."""

from sklearn.metrics import f1_score, roc_auc_score
from .utils import _is_invalid


def compute_auroc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute AUROC. None scores are replaced with the worst possible score for that label."""
    y_scores_clean = []
    for label, score in zip(y_true, y_scores):
        if _is_invalid(score):
            y_scores_clean.append(0.0 if label == 1 else 1.0)
        else:
            y_scores_clean.append(float(score))
    return float(roc_auc_score(y_true, y_scores_clean))


def compute_ovr_auroc(y_true: list[int], y_scores: list[list[float]]) -> float:
    """Compute OvR macro AUROC. y_scores shape (N, C). None rows replaced with worst scores."""
    n_classes = len(y_scores[0])
    y_scores_clean = []
    for label, row in zip(y_true, y_scores):
        if row is None or any(_is_invalid(s) for s in row):
            y_scores_clean.append(
                [0.0 if c == label else 1.0 / (n_classes - 1) for c in range(n_classes)]
            )
        else:
            y_scores_clean.append([float(s) for s in row])
    return float(
        roc_auc_score(y_true, y_scores_clean, multi_class="ovr", average="macro")
    )


def compute_macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    """Compute macro F1. None predictions are replaced with the worst possible prediction."""
    y_pred_clean = []
    for label, pred in zip(y_true, y_pred):
        if _is_invalid(pred):
            y_pred_clean.append(0 if label != 0 else 1)
        else:
            y_pred_clean.append(int(pred))
    return float(f1_score(y_true, y_pred_clean, average="macro", zero_division=0))


def compute_ovr_f1(y_true: list[int], y_pred: list[list[float]]) -> float:
    """Compute OvR macro F1. y_pred shape (N, C). None rows replaced with worst predictions."""
    n_classes = next(len(row) for row in y_pred if row is not None)
    y_pred_clean = []
    for label, row in zip(y_true, y_pred):
        if row is None or any(_is_invalid(s) for s in row):
            y_pred_clean.append((label + 1) % n_classes)
        else:
            y_pred_clean.append(int(max(range(n_classes), key=lambda c: row[c])))
    return float(f1_score(y_true, y_pred_clean, average="macro", zero_division=0))
