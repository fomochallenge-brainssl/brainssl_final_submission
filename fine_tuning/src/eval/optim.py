"""Shared optimizer/scheduler builder for finetuning estimators.

Supports two protocols:

- ``adamw`` + ``multistep`` (default): AdamW with a MultiStepLR schedule
  (milestones at 60% and 80% of training, gamma 0.1). This preserves the
  original behaviour used for AMAES / I-JEPA finetuning.
- ``sgd`` + ``poly``: SGD with Nesterov momentum and a polynomial LR decay
  ``(1 - epoch / max_epochs) ** power``. This mirrors the nnU-Net / nnssl
  downstream finetuning protocol used by CVA (Latent Campus).
"""

from __future__ import annotations

from typing import Optional

import torch


def build_optimizer_and_scheduler(
    parameters,
    *,
    optimizer: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 3e-5,
    momentum: float = 0.99,
    nesterov: bool = True,
    max_epochs: Optional[int] = None,
    lr_schedule: str = "multistep",
    poly_power: float = 0.9,
):
    """Build a Lightning-style optimizer (+ optional scheduler) configuration."""
    optimizer = optimizer.lower()
    lr_schedule = (lr_schedule or "none").lower()

    if optimizer == "sgd":
        opt = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer!r} (use 'adamw' or 'sgd').")

    if max_epochs is None or lr_schedule == "none":
        return [opt]

    if lr_schedule == "poly":
        total = max(int(max_epochs), 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda epoch: (1.0 - min(epoch, total - 1) / total) ** poly_power,
        )
        return [opt], [scheduler]

    if lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[int(max_epochs * 0.6), int(max_epochs * 0.8)],
            gamma=0.1,
        )
        return [opt], [scheduler]

    raise ValueError(
        f"Unsupported lr_schedule {lr_schedule!r} (use 'multistep', 'poly' or 'none')."
    )
