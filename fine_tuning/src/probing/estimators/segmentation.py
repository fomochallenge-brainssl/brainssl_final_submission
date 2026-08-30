##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim

from nidl.estimators.base import BaseEstimator
from nidl.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from src.utils.compute_params_groups import build_param_groups

REGION_LOSSES = ("dice", "focal_tversky")

class DefaultSegModel(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        latent_size: int,
        num_classes: int,
        target_shape: Tuple[int, int, int],
        hidden_channels: int = 128,
    ):
        """Build a lightweight upsampling decoder for classification-style encoders.
        
        Takes a 1-D latent vector of shape ``(B, latent_size)``, reshapes it to a
        spatial volume ``(B, latent_size, 1, 1, 1)``, applies a transposed
        convolution to produce spatial feature maps, then upsamples to
        ``target_shape`` and produces per-voxel class logits.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_size, 1, 1, 1)),
            nn.ConvTranspose3d(latent_size, hidden_channels, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm3d(hidden_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Upsample(size=target_shape, mode="trilinear", align_corners=False),
            nn.Conv3d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def _dice_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Soft multi-class Dice loss.

    Parameters
    ----------
    preds: torch.Tensor
        logits of shape ``(B, C, D, H, W)``.
    targets: torch.Tensor
        integer mask of shape ``(B, D, H, W)``.
    smooth: float
        smoothing constant to avoid division by zero.

    Returns
    -------
    loss: torch.Tensor
        scalar Dice loss averaged over classes and batch.
    """
    num_classes = preds.shape[1]
    probs = func.softmax(preds, dim=1)  # (B, C, D, H, W)
    targets_onehot = func.one_hot(targets.long(), num_classes=num_classes)  # (B, D, H, W, C)
    targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

    intersection = (probs * targets_onehot).sum(dim=(2, 3, 4))
    union = probs.sum(dim=(2, 3, 4)) + targets_onehot.sum(dim=(2, 3, 4))
    dice = (2 * intersection + smooth) / (union + smooth)
    #exclude background (label=0)
    dice = dice[:, 1:]
    return 1.0 - dice.mean()


def _focal_tversky_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.3333,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Focal Tversky loss (Abraham & Khan, ISBI 2019).

    The Tversky index generalises Dice by weighting false negatives and false
    positives separately::

        TI = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

    ``alpha = beta = 0.5`` recovers Dice. Raising ``alpha`` above ``beta``
    penalises *missing* foreground more than over-segmenting it, which is the
    regime for small structures where the model collapses to predicting
    background.

    Parameters
    ----------
    preds: torch.Tensor
        logits of shape ``(B, C, D, H, W)``.
    targets: torch.Tensor
        integer mask of shape ``(B, D, H, W)``.
    alpha: float, default=0.7
        weight of false negatives (missed foreground).
    beta: float, default=0.3
        weight of false positives. Conventionally ``alpha + beta = 1``.
    gamma: float, default=1.3333
        focal exponent applied to ``1 - TI`` as 1/gamma, per sample and class.
    smooth: float
        smoothing constant to avoid division by zero.

    Returns
    -------
    loss: torch.Tensor
        scalar loss, averaged over foreground classes and batch.
    """
    num_classes = preds.shape[1]
    probs = func.softmax(preds, dim=1)
    targets_onehot = func.one_hot(targets.long(), num_classes=num_classes)
    targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

    dims = (2, 3, 4)
    tp = (probs * targets_onehot).sum(dim=dims)
    fn = ((1.0 - probs) * targets_onehot).sum(dim=dims)
    fp = (probs * (1.0 - targets_onehot)).sum(dim=dims)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    # exclude background (label=0), as _dice_loss does
    tversky = tversky[:, 1:]

    return ((1.0 - tversky) ** gamma).mean()


class SegmentationEstimator(BaseEstimator):
    """Segmentation estimator for finetuning pre-trained encoders on voxel-wise
    segmentation tasks.

    Wraps any encoder + decoder composite model. The training objective is a
    compound loss of soft Dice loss and cross-entropy, inspired by nnU-Net v2.
    The decoder architecture is user-defined: any module that maps the encoder
    output to per-voxel class logits of shape ``(B, num_classes, D, H, W)``
    is compatible.

    To freeze the encoder and train only the decoder, use
    :meth:`SegmentationEstimator.freeze_encoder`. It assumes the decoder is
    exposed as a submodule named ``decoder`` in ``model``.

    Parameters
    ----------
    model: nn.Module
        Composite encoder + decoder module. Can use any decoder architecture;
        must expose a ``decoder`` submodule if :meth:`freeze_encoder` is used.
    lr: float
        learning rate for AdamW.
    weight_decay: float
        weight decay for AdamW.
    dice_weight: float, default=0.5
        weight of the Dice term in the compound loss
        ``loss = dice_weight * dice + (1 - dice_weight) * cross_entropy``.
    region_loss: {"dice", "focal_tversky"}, default="dice"
        Which region-overlap term to optimise against cross-entropy.
    tversky_alpha: float, default=0.7
        weight of false negatives; unused unless ``region_loss`` is
        ``focal_tversky``.
    tversky_beta: float, default=0.3
        weight of false positives.
    focal_gamma: float, default=0.75
        focal exponent applied to ``1 - TI``.
    max_epochs: int, default=None
        optionally, use a MultiStepLR scheduler.
    random_state: int, default=None
        seed for reproducibility.
    kwargs: dict
        additional Trainer parameters.

    Attributes
    ----------
    model
        a :class:`~torch.nn.Module` containing the prediction model.
    validation_step_outputs
        a dictionary with the validation predictions and labels.

    Notes
    -----
    A batch of data must contain two elements: an image tensor
    ``(B, C, D, H, W)`` and an integer segmentation mask ``(B, D, H, W)``.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        encoder_lr_ratio: float,
        weight_decay: float,
        dice_weight: float = 0.5,
        region_loss: str = "dice",
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        focal_gamma: float = 1.3333,
        optimizer: str = "adamw",
        momentum: float = 0.99,
        nesterov: bool = True,
        lr_schedule: str = "multistep",
        random_state: Optional[int] = None,
        lr_scheduler_params: Optional[dict] = None,
        **kwargs,
    ):
        if region_loss not in REGION_LOSSES:
            raise ValueError(
                f"region_loss must be one of {REGION_LOSSES}, got {region_loss!r}."
            )
        super().__init__(
            random_state=random_state,
            ignore=["model", "callbacks", "logger"],
            **kwargs,
        )
        self.model = model
        self.lr_scheduler_params = lr_scheduler_params
        self.validation_step_outputs = {}
        self._optimizer = optimizer
        self._momentum = momentum
        self._nesterov = nesterov
        self._lr_schedule = lr_schedule

    def freeze_encoder(self):
        """Freeze the encoder; only the decoder will be trained."""
        print("Freezing encoder's weights")
        self.model.freeze_encoder()

    def unfreeze_encoder(self):
        """Unfreeze the encoder."""
        print("Unfreezing encoder's weights")
        self.model.unfreeze_encoder()

    def configure_optimizers(self):
        """AdamW optimizer with optional MultiStepLR scheduler."""
        params_groups = build_param_groups(
            self.model,
            encoder_lr_ratio=self.hparams.encoder_lr_ratio,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        optimizer = optim.AdamW(params_groups)
        if (
            hasattr(self.hparams, "max_epochs")
            and self.hparams.max_epochs is not None
            and self.hparams.lr_scheduler_params is not None
        ):
            lr_scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                max_epochs=self.hparams.max_epochs,
                **self.hparams.lr_scheduler_params,
            )
            return [optimizer], [lr_scheduler]
        return [optimizer]

    def compound_loss(
        self, batch: Sequence[torch.Tensor], mode: str
    ):
        """Compute compound Dice + cross-entropy loss (nnU-Net v2 style)."""
        imgs, masks, *_ = batch  # ignore meta information when present
        masks = masks.squeeze(1).long()  # (B, D, H, W)
        preds = self.model(imgs)  # (B, num_classes, D, H, W)

        ce = func.cross_entropy(preds, masks)
        # `dice` stays the raw 1 - Dice for logging, whatever is optimised, so
        # *_dice_loss curves compare across region losses.
        dice = _dice_loss(preds, masks)
        if self.hparams.region_loss == "focal_tversky":
            region = _focal_tversky_loss(
                preds, masks,
                alpha=self.hparams.tversky_alpha,
                beta=self.hparams.tversky_beta,
                gamma=self.hparams.focal_gamma,
            )
        else:
            region = dice
        w = self.hparams.dice_weight
        loss = w * region + (1.0 - w) * ce

        if mode in ("train", "val"):
            self.log(mode + "_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(mode + "_dice_loss", dice, prog_bar=False, on_step=False, on_epoch=True)
            self.log(mode + "_ce_loss", ce, prog_bar=False, on_step=False, on_epoch=True)
        return preds, loss, masks

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        _, loss, _ = self.compound_loss(batch, mode="train")
        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        _, loss, _ = self.compound_loss(batch, mode="val")
        print(f"Validation loss: {loss}\n")
        
    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        imgs, *_ = batch
        return self(imgs)