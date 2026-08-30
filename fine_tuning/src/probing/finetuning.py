import os
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore

from src.data.dataset import FOMO26Dataset
from src.data.loader import fomo26_collate
from src.preprocessing import inverse_preprocessing
from src.probing.estimators.linear_regression import LinearRegression
from src.probing.estimators.logistic_regression import LogisticRegression
from src.probing.estimators.segmentation import (
    DefaultSegModel,
    SegmentationEstimator,
)
from src.utils.sliding_window import (
    sliding_window_predict,
)

from .fomo_metrics import (
    compute_absolute_error,
    compute_auroc,
    compute_correlation,
    compute_dice_coefficient,
    compute_macro_f1,
    compute_normalized_surface_distance,
    compute_ovr_auroc,
    compute_ovr_f1,
)

"""
    Get spacing from original representation to compute surface-based metrics
"""
def _spacing_from_affine(affine) -> Tuple[float, float, float]:
    """Per-axis voxel spacing (mm) implied by a RAS affine's column norms."""
    affine = np.asarray(affine, dtype=np.float64)
    return tuple(float(s) for s in np.linalg.norm(affine[:3, :3], axis=0))


def run_one_fold_finetuning(
    encoder: nn.Module,
    dataset: FOMO26Dataset,
    fold: int,
    latent_size: int,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    finetuning_kwargs: dict = None,
    finetuning_callbacks: Optional[Union[list, Callback]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    results_dir: str = "outputs/results/",
    random_state: Optional[int] = 42,
    freeze_encoder: bool = False,
    seg_decoder_config: Optional[DictConfig] = None,
    sliding_window: Optional[dict] = None,
    keep_checkpoint: bool = False,
    inverse_target_transform: Optional[Callable] = None,
    use_last_checkpoint: Optional[bool] = False,
):
    """Finetune a pre-trained encoder on one fold of a FOMO26 task and save metrics.

    Parameters
    ----------
    encoder: nn.Module
        Pre-trained encoder that takes input of shape
        ``(batch_size, n_modalities, H, W, D)`` and outputs a latent vector
        of size ``latent_size``.
    dataset: FOMO26Dataset
        Full dataset with folds defined (via ``folds`` and ``folds_cols``).
    fold: int
        Which fold to evaluate.
    latent_size: int
        Dimensionality of the encoder output vector.
    target_shape: tuple of int, default=(128, 128, 128)
        Spatial size ``(D, H, W)`` used to build the segmentation decoder head.
    finetuning_kwargs: dict, default={"lr": 1e-4}
        Keyword arguments forwarded to the estimator constructor (and thereby
        to the PyTorch Lightning Trainer). Any ``pl.LightningModule`` / Trainer
        parameter can be passed here, e.g. ``lr``, ``weight_decay``,
        ``max_epochs``.
    finetuning_callbacks: list of Callback or Callback, default=None
        Optional PyTorch Lightning callbacks.
    batch_size: int, default=1
    num_workers: int, default=0
    pin_memory: bool, default=False
    results_dir: str, default="outputs/results/"
        Directory where ``metrics.tsv`` will be written.
    random_state: int or None, default=42
    keep_checkpoint: bool, default=False
        By default the best-epoch checkpoint is deleted after computing test metrics.
        Set to ``True`` to keep it instead -- e.g. when training a final model for
        deployment -- in which case the function returns the checkpoint's path.
    use_last_checkpoint: bool, default=False
        Evaluate on the last trained checkpoint instead of best checkpoint.
    """
    if finetuning_kwargs is None:
        finetuning_kwargs = {"lr": 1e-4, "encoder_lr_ratio": 1.0}

    sliding_window_enabled = sliding_window is not None

    train_loader = DataLoader(
        dataset.get_split(fold, split="train",
                          return_raw_labels=False),
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=fomo26_collate,
    )
    if use_last_checkpoint:
        # Disable validation
        val_loader = None
    else:
        val_loader = DataLoader(
            dataset.get_split(
                fold, split="val",
                return_raw_labels=False),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=fomo26_collate,
        )

    test_loader = DataLoader(
        dataset.get_split(
            fold, split="test",
            return_raw_labels=(dataset.task_type == 'segmentation')),
        shuffle=False,
        batch_size=1 if sliding_window_enabled else batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=fomo26_collate,
    )

    task_type = dataset.task_type
    num_classes = dataset.num_classes
    
    if task_type == "classification":
        if use_last_checkpoint:
            print('using last checkpoint')
            # Add checkpointing to retrieve last finetuning epoch
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='last_model',
                monitor=None,  # Saves model for last checkpoint
                every_n_epochs=finetuning_kwargs['check_val_every_n_epoch'],
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
        if not use_last_checkpoint:
            print('using best checkpoint')
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='best',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=False,
            )
        finetuning_callbacks.append(checkpoint_cb)
        finetuning_kwargs['enable_checkpointing'] = True
        model = nn.Sequential(OrderedDict([
            ("encoder", encoder),
            ("fc", nn.Linear(latent_size, num_classes)),
        ]))

        estimator = LogisticRegression(
            model=model,
            num_classes=num_classes,
            random_state=random_state,
            callbacks=finetuning_callbacks,
            **finetuning_kwargs,
        )
        if freeze_encoder:
            estimator.freeze_encoder()
        else:
            estimator.unfreeze_encoder()

        estimator.fit(train_loader, val_loader)
        estimator, ckpt_path = load_checkpoint(estimator, checkpoint_cb, use_last_checkpoint)

        estimator.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(estimator.device)
                logits = estimator.model(imgs)
                probs = torch.softmax(logits, dim=1).detach().cpu()
                y_true.extend(targets)
                y_pred.extend(probs)

        y_true = torch.vstack(y_true).squeeze(-1).tolist()
        y_pred = torch.vstack(y_pred).tolist()  # list of per-class logits
        # Hard predictions (argmax) for F1
        y_pred_labels = [int(max(range(num_classes), key=lambda c: logits[c])) for logits in y_pred]

        # For binary use probability of positive class; for multi-class use OvR
        if num_classes == 2:
            y_scores_pos = [logits[1] for logits in y_pred]
            metrics = {
                "roc_auc_ovr": compute_auroc(y_true, y_scores_pos),
                "f1_macro": compute_macro_f1(y_true, y_pred_labels),
            }
        else:
            metrics = {
                "roc_auc_ovr": compute_ovr_auroc(y_true, y_pred),
                "f1_macro": compute_macro_f1(y_true, y_pred_labels),
            }

    elif task_type == "regression":
        if use_last_checkpoint:
            print('using last checkpoint')
            # Add checkpointing to retrieve last finetuning epoch
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='last_model',
                monitor=None,  # Saves model for last checkpoint
                every_n_epochs=finetuning_kwargs['check_val_every_n_epoch'],
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
        if not use_last_checkpoint:
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='best',
                monitor='val_mae',
                mode='min',
                save_top_k=1,
                save_last=False,
            )
        finetuning_callbacks.append(checkpoint_cb)
        finetuning_kwargs['enable_checkpointing'] = True
        model = nn.Sequential(OrderedDict([
            ("encoder", encoder),
            ("fc", nn.Linear(latent_size, 1)),
        ]))
        estimator = LinearRegression(
            model=model,
            random_state=random_state,
            callbacks=finetuning_callbacks,
            **finetuning_kwargs,
        )

        if freeze_encoder:
            estimator.freeze_encoder()
        else:
            estimator.unfreeze_encoder()

        estimator.fit(train_loader, val_loader)
        estimator, ckpt_path = load_checkpoint(estimator, checkpoint_cb, use_last_checkpoint)
        
        estimator.eval()
        y_true, y_pred = [], []
        # Same values in the space the loss is computed in (z-scored, if a
        # target transform is in use), kept for per-subject inspection.
        y_true_z, y_pred_z = [], []
        with torch.no_grad():
            for imgs, targets in test_loader:
                imgs = imgs.to(estimator.device)
                preds = estimator.model(imgs).squeeze(1).detach().cpu()

                y_true_z.extend(targets.float().tolist())
                y_pred_z.extend(preds.tolist())

                if inverse_target_transform is not None:
                    for target in targets:
                        y_true.append(inverse_target_transform(target.float()))

                    for pred in preds:
                        y_pred.append(inverse_target_transform(pred.float()))
                else:
                    y_true.extend(targets.float().tolist())
                    y_pred.extend(preds.tolist())

        # Per-subject predictions, so that regression behaviour (e.g. collapse
        # onto a group mean) can be inspected outside of aggregate metrics.
        pd.DataFrame({
            "fold": fold,
            "y_true_z": y_true_z,
            "y_pred_z": y_pred_z,
            "y_true": [float(v) for v in y_true],
            "y_pred": [float(v) for v in y_pred],
        }).to_csv(
            os.path.join(results_dir, "predictions.tsv"), sep="\t", index=False
        )

        metrics = {
            "pearson_corrcoef": compute_correlation(y_true=y_true, y_pred=y_pred),
            "mae": compute_absolute_error(y_true=y_true, y_pred=y_pred),
        }

    elif task_type == "segmentation":
        if use_last_checkpoint:
            print('using last checkpoint')
            # Add checkpointing to retrieve last finetuning epoch
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='last_model',
                monitor=None,  # Saves model for last checkpoint
                every_n_epochs=finetuning_kwargs['check_val_every_n_epoch'],
                save_on_train_epoch_end=True,
                save_top_k=1,
            )
        if not use_last_checkpoint:
            checkpoint_cb = ModelCheckpoint(
                dirpath=results_dir,
                filename='best',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                save_last=False,
            )
        finetuning_callbacks.append(checkpoint_cb)
        finetuning_kwargs['enable_checkpointing'] = True

        if seg_decoder_config: 
            decoder_factory = hydra.utils.instantiate(seg_decoder_config)
            model = decoder_factory(
                encoder=encoder,
                num_classes=num_classes
            )

        else:
            # Default decoder that simply upsamples the 1-D latent representation
            # to a 3D volume and computes class logits
            model = DefaultSegModel(encoder, latent_size, num_classes, target_shape)
            

        estimator = SegmentationEstimator(
            model=model,
            random_state=random_state,
            callbacks=finetuning_callbacks,
            **finetuning_kwargs,
        )
        if freeze_encoder:
            estimator.freeze_encoder()
        else:
            estimator.unfreeze_encoder()
        estimator.fit(train_loader, val_loader)
        estimator, ckpt_path = load_checkpoint(estimator, checkpoint_cb, use_last_checkpoint)
        if test_loader.dataset.test_transform is None:
            raise RuntimeError(
                "Segmentation fine-tuning evaluation requires "
                "`dataset.test_transform` (a `src.preprocessing.ForwardPreprocessing` "
                "instance) to be set."
            )

        estimator.eval()
        y_true, y_pred, spacings_mm = [], [], []
        with torch.no_grad():
            if sliding_window_enabled:
                patch_size = sliding_window["patch_size"]
                overlap = sliding_window["overlap"]
                sw_batch_size = sliding_window['sw_batch_size']
                for imgs, labels_preprocessed, metas in test_loader:
                    imgs = imgs.to(estimator.device)
                    meta = metas[0]
                    logits = sliding_window_predict(
                        estimator.model, imgs, patch_size, overlap, num_classes, sw_batch_size
                    )
                    # Invert the per-class logits back
                    # to the native grid, then argmax
                    _, logits_native, _, _ = inverse_preprocessing(
                        image=None, label=logits[0], metas=meta,
                        interpolation_label='linear',
                    )
                    pred = logits_native.argmax(dim=0)  # (D, H, W), native space

                    original_masks = meta['raw_labels']
                    y_pred.append(pred.detach().cpu().long())
                    y_true.append(original_masks[0].int().long())
                    spacings_mm.append(_spacing_from_affine(meta['original_affine']))
            else:
                # dataset returns (imgs, masks, metas) for segmentation
                for imgs, _, metas in test_loader:
                    imgs = imgs.to(estimator.device)
                    logits = estimator.model(imgs)  # (B, C, D', H', W') in preprocessed space
                    # loop through batch elements
                    for b in range(logits.shape[0]):
                        meta = metas[b]
                        # Interpolating discrete class indices is not
                        # meaningful (e.g. averaging class 0 and class 2
                        # yields 1, which need not be the correct boundary
                        # class) -- so invert the per-class logits with
                        # linear interpolation and argmax *after*, in native
                        # space (see src.preprocessing.inverse_resample).
                        _, logits_native, _, _ = inverse_preprocessing(
                            image=None, label=logits[b], metas=meta,
                            interpolation_label='linear',
                        )
                        # at each voxel, the highest-probability class is used as prediction
                        pred = logits_native.argmax(dim=0)  # (D, H, W), native space

                        y_pred.append(pred.unsqueeze(0).detach().cpu().long())
                        y_true.append(meta['raw_labels'][0].int().long())  # long type is necessary for DiceScore
                        spacings_mm.append(_spacing_from_affine(meta['original_affine']))

        # Compute Dice and NSD
        # If binary segmentation, use FOMO26 implementation
        if num_classes == 2:
            dice_scores, nsd_scores = [], []
            for pred, label, spacing_mm in zip(y_pred, y_true, spacings_mm):
                pred_mask = pred.squeeze(0).numpy()
                gt_mask = label.int().squeeze(0).numpy()
                dice_scores.append(compute_dice_coefficient(pred_mask=pred_mask, gt_mask=gt_mask))
                nsd_scores.append(compute_normalized_surface_distance(
                    pred_mask=pred_mask, gt_mask=gt_mask, spacing_mm=spacing_mm))
            metrics = {
                "dice": float(np.mean(dice_scores)),
                "nsd": float(np.mean(nsd_scores)),
            }

        # If multi-class then FOMO26 implementation not available yet, use torchmetrics
        if num_classes > 2:
            dice = DiceScore(num_classes=num_classes, include_background=False, average="macro",
                            aggregation_level="samplewise", input_format='index')
            for pred, label in zip(y_pred, y_true):
                dice(pred.unsqueeze(0), label.unsqueeze(0)) # DiceScore expects to have batch size as first dimension

            metrics = {"dice": dice.compute().item()}
            dice.reset()

    else:
        raise ValueError(
            'task_type must be one of ["classification", "regression", '
            f'"segmentation"], got "{task_type}".'
        )

    # Save metrics
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    results_file = os.path.join(results_dir, "metrics.tsv")
    rows = [[fold, name, _to_float(value)] for name, value in metrics.items()]
    pd.DataFrame(rows, columns=["fold", "metric", "value"]).to_csv(
        results_file, sep="\t", index=False
    )

    if keep_checkpoint:
        return ckpt_path

    # Delete checkpoint
    Path(ckpt_path).unlink(missing_ok=False)


def load_checkpoint(estimator, checkpoint_cb, use_last_checkpoint):
    '''Loads either best epoch or last epoch (depending on ``use_last_checkpoint`` value.)'''
    ckpt_path = checkpoint_cb.best_model_path
    if not ckpt_path:
        raise RuntimeError(
            "No checkpoint was saved (check checkpoint saving frequency)."
        )
    if use_last_checkpoint:
        # No need to reload state_dict, estimator stays unchanged
        return estimator, ckpt_path
    
    ckpt = torch.load(ckpt_path, map_location=estimator.device, weights_only=False)
    estimator.load_state_dict(ckpt["state_dict"])
    # Estimator is moved to cpu after fit. Need to retrieve correct device
    device = estimator.trainer.strategy.root_device
    if estimator.trainer.num_devices > 1:
        warnings.warn(
            f"Trainer was configured with {estimator.trainer.num_devices} devices; "
            f"test-time inference here runs on a single device only "
            f"({device}) -- the rest are unused."
        )
    estimator = estimator.to(device)
    return estimator, ckpt_path