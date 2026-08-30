"""Linear regression finetuning estimator with configurable optimizer/scheduler."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as func
from torch import nn

from nidl.estimators.base import BaseEstimator, RegressorMixin

from src.eval.optim import build_optimizer_and_scheduler


class LinearRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        optimizer: str = "adamw",
        momentum: float = 0.99,
        nesterov: bool = True,
        lr_schedule: str = "multistep",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(random_state=random_state, ignore=["model"], **kwargs)
        self.model = model
        self.validation_step_outputs = {}
        self._optimizer = optimizer
        self._momentum = momentum
        self._nesterov = nesterov
        self._lr_schedule = lr_schedule

    def freeze_encoder(self):
        self.model.requires_grad_(False)
        self.model.fc.requires_grad_(True)

    def configure_optimizers(self):
        return build_optimizer_and_scheduler(
            self.parameters(),
            optimizer=self._optimizer,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=self._momentum,
            nesterov=self._nesterov,
            max_epochs=getattr(self.hparams, "max_epochs", None),
            lr_schedule=self._lr_schedule,
        )

    def mse_loss(self, batch: Sequence[torch.Tensor], mode: str):
        imgs, targets = batch
        targets = targets.float()
        preds = self.model(imgs).squeeze(1)
        loss = func.mse_loss(preds, targets)
        mae = (preds - targets).abs().mean()
        if mode in ("train", "val"):
            self.log(mode + "_loss", loss, prog_bar=True)
            self.log(mode + "_mae", mae, prog_bar=True)
        return preds, loss, targets

    def training_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        _, loss, _ = self.mse_loss(batch, mode="train")
        return loss

    def validation_step(
        self,
        batch: Sequence[torch.Tensor],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        preds, _, targets = self.mse_loss(batch, mode="val")
        self.validation_step_outputs.setdefault("pred", []).append(preds)
        self.validation_step_outputs.setdefault("label", []).append(targets)

    def on_validation_epoch_end(self):
        self.validation_step_outputs.clear()

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ):
        return self.model(batch).squeeze(1)
