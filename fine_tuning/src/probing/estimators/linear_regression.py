##########################################################################
# NSAp - Copyright (C) CEA, 2025
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as func
from torch import nn, optim

from nidl.estimators.base import BaseEstimator, RegressorMixin
from nidl.utils.lr_scheduler import LinearWarmupCosineAnnealingLR

from src.utils.compute_params_groups import build_param_groups

LOSSES = ("mse", "l1", "focal_mse", "focal_l1")
FOCAL_ACTIVATIONS = ("sigmoid", "tanh")


def focal_r_scaling(
    errors: torch.Tensor,
    beta: float = 0.2,
    gamma: float = 1.0,
    activate: str = "sigmoid",
) -> torch.Tensor:
    """Continuous focusing factor of the Focal-R loss, valued in ``[0, 1)``.

    Parameters
    ----------
    errors: torch.Tensor
        Signed per-sample errors ``pred - target``. Only the magnitude is used.
    beta: float, default=0.2
        Steepness of the mapping. 
    gamma: float, default=1.0
        Focusing exponent. ``gamma=0`` disables focusing and recovers the plain
        L1/MSE loss; larger values concentrate the loss on hard samples.
    activate: {"sigmoid", "tanh"}, default="sigmoid"
        Activation function applied to ``beta * |error|``.

    Returns
    -------
    torch.Tensor
        Per-sample scaling factors, same shape as ``errors``.
    """
    if activate not in FOCAL_ACTIVATIONS:
        raise ValueError(
            f"activate must be one of {FOCAL_ACTIVATIONS}, got {activate!r}."
        )
    abs_errors = beta * errors.abs()
    if activate == "sigmoid":
        factor = 2 * torch.sigmoid(abs_errors) - 1
    else:
        factor = torch.tanh(abs_errors)
    return factor**gamma


def focal_r_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    base: str = "l1",
    beta: float = 0.2,
    gamma: float = 1.0,
    activate: str = "sigmoid",
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal-R loss ``mean_i sigma(|beta * e_i|)^gamma * e_i``.

    Parameters
    ----------
    preds, targets: torch.Tensor
        Predictions and continuous targets, same shape.
    base: {"l1", "mse"}, default="l1"
        Error term ``e_i`` the focusing factor multiplies. ``l1`` is the
        formulation given in the paper; ``mse`` is the squared variant shipped
        in the authors' reference implementation.
    beta, gamma, activate:
        Forwarded to :func:`focal_r_scaling`.
    weights: torch.Tensor, default=None
        Optional per-sample weights, e.g. the inverse of an LDS-smoothed label
        density. Can be used for combining Focal-R with LDS.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if base not in ("l1", "mse"):
        raise ValueError(f"base must be 'l1' or 'mse', got {base!r}.")
    errors = preds - targets
    loss = errors.abs() if base == "l1" else errors**2
    loss = loss * focal_r_scaling(errors, beta=beta, gamma=gamma, activate=activate)
    if weights is not None:
        loss = loss * (weights / weights.mean())
    return loss.mean()

class LinearRegression(RegressorMixin, BaseEstimator):
    """Linear regression estimator for finetuning pre-trained encoders on
    continuous target prediction tasks (e.g. brain age estimation).

    After training an encoder via self-supervised learning, this class can be
    used to probe or finetune on downstream regression tasks by learning a
    linear mapping from representations to continuous predictions. To freeze
    the encoder, consider using :meth:`LinearRegression.freeze_encoder`. It
    assumes the linear layer is named ``fc``.

    Examples
    --------
    >>> model = nn.Sequential(OrderedDict([
    >>>     ("encoder", encoder),
    >>>     ("fc", nn.Linear(latent_size, 1))
    >>> ]))
    >>> regressor = LinearRegression(model=model, lr=1e-4, weight_decay=1e-4)
    >>> regressor.fit(train_loader, val_loader)

    Parameters
    ----------
    model: nn.Module
        the encoder f(.) architecture. Must expose a ``fc`` submodule
        (a linear layer) as the regression head.
    lr: float
        the learning rate for the AdamW optimizer.
    weight_decay: float
        the weight decay parameter for the AdamW optimizer.
    max_epochs: int, default=None
        optionally, use a MultiStepLR scheduler.
    random_state: int, default=None
        seed for reproducibility.
    loss: {"mse", "l1", "focal_mse", "focal_l1"}, default="mse"
        Training criterion. The ``focal_*`` variants are the Focal-R loss.
    focal_beta: float, default=0.2
        ``beta`` of the Focal-R scaling factor, in units of ``1/target``.
        Unused unless ``loss`` is a ``focal_*`` variant.
    focal_gamma: float, default=1.0
        ``gamma`` of the Focal-R scaling factor.
    focal_activate: {"sigmoid", "tanh"}, default="sigmoid"
        Mapping from absolute error to ``[0, 1)``.
    kwargs: dict
        additional Trainer parameters passed to
        :class:`~nidl.estimators.base.BaseEstimator`.

    Attributes
    ----------
    model
        a :class:`~torch.nn.Module` containing the prediction model.
    validation_step_outputs
        a dictionary with the validation predictions and associated labels
        in the ``'pred'`` and ``'label'`` keys respectively.

    Notes
    -----
    A batch of data must contain two elements: a tensor with images and a
    tensor with the continuous target values.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        encoder_lr_ratio: float,
        weight_decay: float,
        random_state: Optional[int] = None,
        lr_scheduler_params: Optional[dict] = None,
        loss: str = "mse",
        focal_beta: float = 0.2,
        focal_gamma: float = 1.0,
        focal_activate: str = "sigmoid",
        **kwargs,
    ):
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}.")
        if focal_activate not in FOCAL_ACTIVATIONS:
            raise ValueError(
                f"focal_activate must be one of {FOCAL_ACTIVATIONS}, "
                f"got {focal_activate!r}."
            )
        super().__init__(random_state=random_state, ignore=["model"], **kwargs)
        self.model = model
        self.lr_scheduler_params = lr_scheduler_params
        self.validation_step_outputs = {}

    def freeze_encoder(self):
        """Freeze the input encoder. Useful for self supervised settings."""
        self.model.requires_grad_(False)
        self.model.fc.requires_grad_(True)

    def unfreeze_encoder(self):
        """Unfreeze the input encoder."""
        self.model.requires_grad_(True)
        self.model.fc.requires_grad_(True)

    def configure_optimizers(self):
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

    def compute_loss(self, batch: Sequence[torch.Tensor], mode: str):
        """Compute and log the configured training criterion.

        ``mode + "_loss"`` is whatever ``loss`` selects, so checkpointing on
        ``val_loss`` keeps selecting on the criterion actually optimised.
        ``mode + "_mae"`` is always logged unweighted, in target units, and is
        the metric to compare across different ``loss`` settings.
        """
        imgs, targets = batch
        targets = targets.float()
        preds = self.model(imgs).squeeze(1)
        name = self.hparams.loss
        if name == "mse":
            loss = func.mse_loss(preds, targets)
        elif name == "l1":
            loss = func.l1_loss(preds, targets)
        else:
            loss = focal_r_loss(
                preds,
                targets,
                base="l1" if name == "focal_l1" else "mse",
                beta=self.hparams.focal_beta,
                gamma=self.hparams.focal_gamma,
                activate=self.hparams.focal_activate,
            )
        mae = (preds - targets).abs().mean()
        if mode in ("train", "val"):
            self.log(mode + "_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(mode + "_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return preds, loss, targets

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        _, loss, _ = self.compute_loss(batch, mode="train")
        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        preds, _, targets = self.compute_loss(batch, mode="val")
        self.validation_step_outputs.setdefault("pred", []).append(preds)
        self.validation_step_outputs.setdefault("label", []).append(targets)

    def on_validation_epoch_end(self):
        """Clean the validation cache at each epoch end."""
        self.validation_step_outputs.clear()

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        return self.model(batch).squeeze(1)