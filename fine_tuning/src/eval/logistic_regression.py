"""LogisticRegression finetuning estimator with a configurable optimizer.

Thin subclass of :class:`nidl.estimators.linear.LogisticRegression` that keeps
the original training/validation logic but lets the optimizer and LR schedule
be selected (AdamW+MultiStepLR by default, or SGD+poly for the CVA protocol).
"""

from __future__ import annotations

from nidl.estimators.linear import LogisticRegression

from src.eval.optim import build_optimizer_and_scheduler


class FinetuneLogisticRegression(LogisticRegression):
    def __init__(
        self,
        *args,
        optimizer: str = "adamw",
        momentum: float = 0.99,
        nesterov: bool = True,
        lr_schedule: str = "multistep",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._optimizer = optimizer
        self._momentum = momentum
        self._nesterov = nesterov
        self._lr_schedule = lr_schedule

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
