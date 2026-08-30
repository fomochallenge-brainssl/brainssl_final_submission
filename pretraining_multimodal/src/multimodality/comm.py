"""
CoMM as an nidl estimator. Rewrite of the original CoMM LightningModule
as nidl's `BaseEstimator`.

[1] What to align in multimodal contrastive learning,
    Dufumier & Castillo-Navarro et al., ICLR 2025
"""
from collections import OrderedDict
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from torch.optim import Optimizer

from nidl.estimators.base import BaseEstimator, TransformerMixin
from nidl.estimators.ssl.utils.optimizer import configure_ssl_optimizers

from src.multimodality.mmfusion import MissingModMMFusion
from src.multimodality.input_adapters import SimpleFeaturesInputAdapter
from src.multimodality.comm_loss import MaskedCoMMLoss


PLAIN_HPARAM_TYPES = (bool, int, float, str, bytes, type(None))


def is_plain_data(value: Any, depth: int = 0) -> bool:
    """True for scalars and for containers holding only scalars.

    Anything else is a live runtime object (a module, a logger, a strategy, a
    callback) that must not end up in the checkpoint's `hyper_parameters`.
    """
    if isinstance(value, PLAIN_HPARAM_TYPES):
        return True
    if depth >= 6:
        return False
    if isinstance(value, (list, tuple, set)):
        return all(is_plain_data(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, PLAIN_HPARAM_TYPES) and is_plain_data(item, depth + 1)
                   for key, item in value.items())
    return False


class CoMM(TransformerMixin, BaseEstimator):
    """CoMM over a single shared encoder or multiple per modality encoders.

    Parameters
    ----------
    encoder : nn.Module or sequence of nn.Module
        Backbone applied to every modality, or several backbones (e.g. one for
        the structural modalities and one for diffusion) dispatched by
        `mod_to_encoder`. All backbones must output feature maps of the same
        shape, since a single `adapter` config tokenizes them.
    num_modalities : int
        Number of modality slots M (matches the dataset's channel order).
    mod_to_encoder : sequence of int or None, default=None
        Length-M index of the backbone encoding each modality slot. Required
        when `encoder` holds several backbones; `None` means one shared
        backbone for every slot.
    adapter : nn.Module or None, default=None
        Tokenizer for the encoder output. Defaults to
        `SimpleFeaturesInputAdapter` (pooled vector -> 1 token per modality);
        `Patched3DInputAdapter` tokenizes a feature map instead.
    enc_ckpt : str, sequence of str or None
        Path to a CNN Lightning checkpoint; its `encoder.` weights are loaded.
        With several backbones, either one shared path or one per backbone
        (`None` entries leave that backbone at its initialization).
    freeze_encoder : bool, default=True
        Freeze the backbone (no grad, eval mode) and train only fusion + head.
    per_modality_adapter : bool, default=False
        Give each modality slot its own copy of `adapter` rather than sharing
        one
    embed_dim : int, default=2048
        Fusion width. Must equal the adapter's token dim.
    fusion, pool, n_heads, n_layers, add_bias_kv, dropout
        `FusionTransformer` hyper-parameters.
    proj_hidden_dim, proj_output_dim : int
        Projection head dims (its input dim is `embed_dim`).
    temperature : float, default=0.1
        CoMM/InfoNCE temperature.
    subset_weights : sequence of float or None
        Optional per-subset loss weights (length M+1: one per modality + full).
    gather_loss : bool, default=False
        All-gather subset embeddings across GPUs before the InfoNCE so each
        subset's negatives span the global batch.
    optimizer, learning_rate, weight_decay, exclude_bias_and_norm_wd,
    optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs
        Optimization settings (same semantics as nidl's SimCLR).
    **kwargs
        Trainer options forwarded to `BaseEstimator` (max_epochs, strategy,
        devices, precision, callbacks, logger, ...).

    Attributes
    ----------
    encoder : MissingModMMFusion
        Multi-modal fusion encoder (backbone(s) + adapter + fusion).
    head : torch.nn.Module
        Projector that maps the fused embedding to the latent space.
    loss : MaskedCoMMLoss
        CoMM loss, restricted to the observed modalities.
    """

    def __init__(
        self,
        encoder: Union[nn.Module, Sequence[nn.Module]],
        num_modalities: int,
        mod_to_encoder: Optional[Sequence[int]] = None,
        adapter: Optional[nn.Module] = None,
        enc_ckpt: Optional[Union[str, Sequence[Optional[str]]]] = None,
        freeze_encoder: bool = True,
        per_modality_adapter: bool = False,
        embed_dim: int = 2048,
        fusion: str = "concat",
        pool: str = "cls",
        n_heads: int = 8,
        n_layers: int = 1,
        add_bias_kv: bool = False,
        dropout: float = 0.0,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 256,
        temperature: float = 0.1,
        subset_weights: Optional[Sequence[float]] = None,
        gather_loss: bool = False,
        optimizer: Union[str, Optimizer, type] = "adamW",
        learning_rate: float = 3e-4,
        weight_decay: float = 5e-4,
        exclude_bias_and_norm_wd: bool = True,
        optimizer_kwargs: Optional[dict] = None,
        lr_scheduler: Optional[str] = "warmup_cosine",
        lr_scheduler_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        #to prevent errors when loading after ddp strategy find_unused_parameters true
        ignore = list(kwargs.pop("ignore", []))
        ignore += [name for name in ("callbacks", "encoder", "adapter", "enc_ckpt")
                   if name not in ignore]
        ignore += [name for name, value in kwargs.items()
                   if name not in ignore and not is_plain_data(value)]
        super().__init__(**kwargs, ignore=ignore)
        self._drop_object_hparams()

        # multi-modal encoder: backbone(s) + tokenizer + latent fusion
        self.encoder = MissingModMMFusion(
            encoder=encoder,
            adapter=adapter if adapter is not None else SimpleFeaturesInputAdapter(),
            num_modalities=num_modalities,
            mod_to_encoder=mod_to_encoder,
            embed_dim=embed_dim,
            enc_ckpt=enc_ckpt,
            freeze_encoder=freeze_encoder,
            per_modality_adapter=per_modality_adapter,
            fusion=fusion,
            pool=pool,
            n_heads=n_heads,
            n_layers=n_layers,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
        )

        # build a 3-layers projector
        # Usual projection head of contrastive learning to avoid considering
        # only pre-training-task-specific properties of the representation
        self.head = self._build_mlp(embed_dim, proj_hidden_dim, proj_output_dim)

        # Build the loss
        self.loss = MaskedCoMMLoss(temperature=temperature, weights=subset_weights, gather=gather_loss)

        # optimization config
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm_wd = exclude_bias_and_norm_wd
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self._fill_default_lr_scheduler_kwargs()

    def _drop_object_hparams(self):
        """keep only plain-data hyper-parameters. this catches anything else
        a caller hands over as an object, so the checkpoint stays loadable with
        torch's default `weights_only=True`.
        """
        dropped = sorted(key for key, value in self.hparams.items()
                         if not is_plain_data(value))
        initial = getattr(self, "_hparams_initial", None)
        for key in dropped:
            self._hparams.pop(key, None)
            if initial is not None:
                initial.pop(key, None)
        if dropped:
            print(f"[CoMM] dropped object-valued hyper-parameters: {dropped}")

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        # BatchNorm1d instead of SyncBatchNorm.
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, batch):
        """Encode both augmentations of a multimodal batch.

        The fusion encoder returns one embedding per modality subset (the
        prototype, i.e. all present modalities, last) plus the presence mask that
        the masked loss needs.
        """
        sample_idx, mod_idx = batch["sample_idx"], batch["mod_idx"]
        B = batch["num_subjects"]

        # compute features for all modality subsets
        # output structure (k single-modality embeddings + 1 all-modalities embedding)
        z1, present = self.encoder(batch["aug1"], sample_idx, mod_idx, B)
        # all-modalities embedding same as before --> discard it
        z2, _ = self.encoder(batch["aug2"], sample_idx, mod_idx, B)
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        return {"aug1_embed": z1,
                "aug2_embed": z2,
                "prototype": -1,
                "present": present}

    def _shared_step(self, batch: Sequence[Any], is_train: bool = True):
        """Shared code for training and validation steps."""
        return self.loss(self(batch))

    def _log_modality_occupancy(self, batch):
        """How often each modality slot is empty on a rank.
        """
        num_mod = self.encoder.num_modalities
        counts = torch.bincount(batch["mod_idx"], minlength=num_mod).float()
        for m in range(num_mod):
            self.log(f"data/n_scans_mod{m}", counts[m],
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"data/absent_mod{m}", (counts[m] == 0).float(),
                     on_step=False, on_epoch=True, sync_dist=True)

    def _log_subsets(self, outputs, stage: str):
        """Per-subset loss/accuracy: modality slot i, then the prototype last.
        """
        for key, value in outputs.items():
            if key.startswith(("ssl_acc_", "ssl_loss_")):
                self.log(f"{key}/{stage}", value,
                         on_step=False, on_epoch=True, sync_dist=True)

    def training_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Perform one training step and compute the training loss.

        Returns
        -------
        outputs : dict
            The CoMM loss dict: "loss", "ssl_acc", and the per-subset
            "ssl_loss_i" / "ssl_acc_i".
        """
        self._log_modality_occupancy(batch)
        outputs = self._shared_step(batch, is_train=True)
        self.log("loss/train", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("ssl_acc/train", outputs["ssl_acc"], prog_bar=True, sync_dist=True)
        self.log("loss/train_epoch", outputs["loss"],
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log("ssl_acc/train_epoch", outputs["ssl_acc"],
                 on_step=False, on_epoch=True, sync_dist=True)
        self._log_subsets(outputs, "train")
        return outputs

    def validation_step(
        self,
        batch: Sequence[Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Perform one validation step and compute the validation loss."""
        outputs = self._shared_step(batch, is_train=False)
        self.log("loss/val", outputs["loss"], prog_bar=True, sync_dist=True)
        self.log("ssl_acc/val", outputs["ssl_acc"], prog_bar=True, sync_dist=True)
        self._log_subsets(outputs, "val")
        return outputs

    def test_step(self, batch, batch_idx):
        return

    def transform_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Encode the input data into the latent space.
        (discard projection head).
        """
        z, _ = self.encoder(batch["aug1"], batch["sample_idx"], batch["mod_idx"],
                            batch["num_subjects"])
        return z[-1]        # the prototype: all present modalities fused

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler in CoMM."""
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        params = [
            {"name": "backbone", "params": backbone_params},
            {"name": "head", "params": self.head.parameters()},
        ]
        return configure_ssl_optimizers(
            trainer=self.trainer,
            optim_params=params,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            exclude_bias_and_norm_wd=self.exclude_bias_and_norm_wd,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_kwargs=self.lr_scheduler_kwargs,
        )

    def _fill_default_lr_scheduler_kwargs(self):
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}

        self.lr_scheduler_kwargs.setdefault("warmup_epochs", 10)
        self.lr_scheduler_kwargs.setdefault("interval", "step")
        self.lr_scheduler_kwargs.setdefault("warmup_start_lr", 1e-6)
        self.lr_scheduler_kwargs.setdefault("min_lr", 0.0)