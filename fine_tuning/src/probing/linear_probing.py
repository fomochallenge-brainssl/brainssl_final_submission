import os
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from nidl.estimators import BaseEstimator
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_validate
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import FOMO26Dataset
from ..estimators.wrapper import EstimatorWrapper
from .scorers import f1_scorer, roc_auc_scorer


def _pearson_corrcoef(y_true, y_pred):
    # pearsonr returns (r, p-value); keep only r
    return pearsonr(y_true, y_pred)[0]

def run_linear_probing(
    encoder: nn.Module,
    dataset: FOMO26Dataset,
    save_embeddings: bool = True,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    results_dir: str = "outputs/results/",
    estimator_kwargs: Optional[dict] = None,
):
    """Compute embeddings for a dataset and run linear probing with cross-validation."""

    task_type = dataset.task_type
    if task_type == "segmentation":
        raise NotImplementedError("Linear probing for segmentation is not implemented yet.")

    # Initialize estimator_kwargs defaut with empty dict
    if not estimator_kwargs:
        estimator_kwargs = {}

    # Wrap encoder if not already a nidl estimator
    if not isinstance(encoder, BaseEstimator):
        estimator = EstimatorWrapper(encoder, is_fitted=True, **estimator_kwargs)
    else:
        estimator = encoder

    # We assume the dataset has no duplicated samples
    assert not dataset.has_duplicates(), 'For linear probing, the provided dataset should have only unique samples as ' \
        f'defined by the columns `folds_cols`, in this case columns {dataset.folds_cols}.'
    # Remove data augmentation for linear probing
    dataset.remove_data_augmentation()

    loader_all = DataLoader(
        dataset,
        shuffle=False,  # must be False to preserve fold index alignment
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    embeddings, labels = estimator.transform_with_targets(loader_all)
    embeddings = embeddings.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    os.makedirs(results_dir, exist_ok=True)

    if save_embeddings:
        emb_file = os.path.join(results_dir, "embeddings.npy")
        with open(emb_file, "wb") as f:
            np.save(f, embeddings)

    cv = dataset.get_folds_indices()  # Can have duplicated samples in trainval

    if task_type == "classification":
        clf = LogisticRegressionCV(class_weight='balanced', max_iter=1000, refit=True,
                                   cv=StratifiedGroupKFold(n_splits=5))
        scoring = {"roc_auc_ovr": roc_auc_scorer, "f1_macro": f1_scorer}
    elif task_type == "regression":
        clf = RidgeCV(cv=GroupKFold(n_splits=5))
        pearson_scorer = make_scorer(_pearson_corrcoef, greater_is_better=True)
        scoring = {
            "neg_mae": "neg_mean_absolute_error",
            "pearson_corrcoef": pearson_scorer,
        }

    # metadata routing to pass groups to inner cv so no overlap in train/val
    # groups on sampled id is necessary here because there can be duplicated
    # samples in trainval splits of cv (bootstrapped folds).
    sklearn.set_config(enable_metadata_routing=True)
    cv_results = cross_validate(clf, X=embeddings, y=labels, cv=cv,
                                scoring=scoring,
                                params={'groups':np.arange(len(embeddings))})

    rows = []
    for metric in scoring:
        for fold, value in enumerate(cv_results[f"test_{metric}"], start=0):
            if metric.startswith('neg'):
                value = -value
                metric_name = metric.replace('neg_', '')
            else:
                metric_name = metric
            rows.append({"fold": fold, "metric": metric_name, "value": value})

    results_file = os.path.join(results_dir, "metrics.tsv")
    pd.DataFrame(rows).to_csv(results_file, sep="\t", index=False)
