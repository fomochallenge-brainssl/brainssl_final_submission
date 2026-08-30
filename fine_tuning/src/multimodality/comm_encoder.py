"""
Eval-side wrapper for a pretrained CoMM fusion encoder.
"""

from typing import Optional, Sequence

import torch
from torch import nn


class CoMMEncoder(nn.Module):
    """Adapt a CoMM ``MissingModMMFusion`` to the dense probing interface.

    Returns the fused prototype embedding (all present modalities), matching
    CoMM's own ``transform_step``.

    Parameters
    ----------
    fusion : nn.Module
        A pretrained ``MissingModMMFusion`` (``model.encoder`` of a loaded CoMM).
    mod_slots : sequence of int, optional
        Pretraining slot index for each downstream channel, in channel order.
        Defaults to ``range(n_modalities)``.
    """

    def __init__(self, fusion: nn.Module, mod_slots: Optional[Sequence[int]] = None):
        super().__init__()
        self.fusion = fusion
        self.mod_slots = None if mod_slots is None else list(mod_slots)

    def freeze_backbone(self, freeze: bool = True) -> "CoMMEncoder":
        """(Un)freeze the CNN backbones; the fusion transformer stays trainable.

        With ``freeze=True`` only the fusion/attention head is finetuned
        on top of fixed pretrained backbones. 
        """
        backbones = self.fusion.encoders
        for p in backbones.parameters():
            p.requires_grad_(not freeze)
        self.fusion.freeze_encoder = freeze
        backbones.eval() if freeze else backbones.train()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n, H, W, D) — one channel per modality.
        B, n, H, W, D = x.shape
        num_mod = getattr(self.fusion, "num_modalities", n)

        if self.mod_slots is None:
            slots = list(range(n))
        else:
            slots = self.mod_slots
            if len(slots) != n:
                raise ValueError(
                    f"mod_slots has {len(slots)} entries but the task provides "
                    f"{n} modalities; they must match (one slot per channel)."
                )
        if max(slots) >= num_mod or min(slots) < 0:
            raise ValueError(
                f"Modality slots {slots} are out of range for a CoMM fusion "
                f"pretrained with num_modalities={num_mod} (valid: 0..{num_mod - 1})."
            )

        # Pack modalities into a flat (B*n, 1, H, W, D) batch, subject-major:
        # [(s0,m0), (s0,m1), ..., (s1,m0), ...].
        packed = x.reshape(B * n, 1, H, W, D)
        sample_idx = torch.arange(B, device=x.device).repeat_interleave(n)
        mod_idx = torch.tensor(slots, device=x.device, dtype=torch.long).repeat(B)

        z, _present = self.fusion(packed, sample_idx, mod_idx, B)
        return z[-1]  # prototype: all present modalities fused, (B, embed_dim)