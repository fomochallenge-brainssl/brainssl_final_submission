from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence


def compute_max_disparity(
    y_true: Sequence[int],
    y_scores: Sequence,
    groups: Sequence,
    metric_fn: Callable,
) -> float:
    """Max - min of ``metric_fn`` evaluated within each group of ``groups``.

    Groups are the fairness-variable values for each sample. A group is dropped
    when ``metric_fn`` raises or returns NaN.

    Return value:
    - ``NaN`` if **zero** groups have a valid metric.
    - ``0.0`` if **exactly one** group has a valid metric.
    - ``max - min`` of the valid per-group values otherwise.
    """
    if not (len(y_true) == len(y_scores) == len(groups)):
        raise ValueError("y_true, y_scores, groups must have the same length")

    bucketed: dict[Any, tuple[list, list]] = {}
    for label, score, grp in zip(y_true, y_scores, groups):
        if grp is None:
            continue
        yt, ys = bucketed.setdefault(grp, ([], []))
        yt.append(label)
        ys.append(score)

    per_group_values: list[float] = []
    for yt, ys in bucketed.values():
        try:
            value = float(metric_fn(yt, ys))
        except Exception:
            continue
        if math.isnan(value):
            continue
        per_group_values.append(value)

    if not per_group_values:
        return float("nan")
    if len(per_group_values) == 1:
        return 0.0
    return max(per_group_values) - min(per_group_values)


def compute_fairness_score(
    y_true: Sequence[int],
    y_scores: Sequence,
    groups_by_variable: Mapping[str, Sequence],
    metric_fn: Callable,
) -> dict:
    """Fairness score for a single metric, aggregated across fairness variables.

    Implements

        FairnessScore(M) = (1 / |V'|) * Σ_{v in V'} (1 - D_v(M))

    where ``D_v(M)`` is :func:`compute_max_disparity` and ``V'`` is the subset
    of fairness variables whose disparity is defined. Variables with undefined
    ``D_v`` are excluded from the average rather than treated as zero-disparity.
    """
    disparities: dict[str, float] = {}
    contributions: dict[str, float] = {}
    for var_name, groups in groups_by_variable.items():
        if len(groups) != len(y_true):
            raise ValueError(
                f"groups for '{var_name}' has length {len(groups)} but y_true has {len(y_true)}"
            )
        valid = [(t, s, g) for t, s, g in zip(y_true, y_scores, groups) if g is not None]
        if valid:
            yt, ys, gs = zip(*valid)
            d = compute_max_disparity(list(yt), list(ys), list(gs), metric_fn)
        else:
            d = float("nan")
        disparities[var_name] = d
        if not math.isnan(d):
            contributions[var_name] = 1.0 - d

    score = sum(contributions.values()) / len(contributions) if contributions else float("nan")
    return {
        "score": score,
        "disparities": disparities,
        "variables_used": list(contributions.keys()),
        "per_variable_contribution": contributions,
    }
