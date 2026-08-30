"""Adapters for using single-modality pre-trained encoders with multi-modal inputs.

When a pre-trained encoder was trained on single-channel (e.g. T1-only) data but
the downstream task provides N > 1 modalities as separate channels, the encoder's
stem (first Conv3d layer) must be adapted to accept N input channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _set_stem_conv(parent: nn.Module, attr: str, new_conv: nn.Conv3d) -> None:
    """Replace a stem Conv3d, including inside ConvDropoutNormNonlin wrappers.

    gardening_tools blocks keep the conv both as ``self.conv`` and as the first
    entry of ``self.all_modules``; forward uses ``all_modules``, so both must
    be updated.
    """
    setattr(parent, attr, new_conv)
    if hasattr(parent, "all_modules") and isinstance(parent.all_modules, nn.Sequential):
        children = list(parent.all_modules.children())
        if children:
            children[0] = new_conv
            parent.all_modules = nn.Sequential(*children)


def adapt_encoder_to_n_modalities(encoder: nn.Module, n_modalities: int) -> nn.Module:
    """Adapt a single-channel pre-trained encoder to accept ``n_modalities`` input channels.

    Locates the first :class:`~torch.nn.Conv3d` layer in the encoder (the stem),
    asserts that it currently accepts exactly 1 input channel, then replaces it
    with a new ``Conv3d`` whose weights are initialised by repeating the original
    weights ``n_modalities`` times along the channel dimension and dividing by
    ``n_modalities``. This preserves the expected activation magnitude at
    initialisation while giving each modality equal contribution.

    The modification is performed **in-place**: the encoder's stem layer is
    replaced inside the parent module.

    Inspired by the weight repetition strategy in
    https://github.com/Sllambias/asparagus.

    Parameters
    ----------
    encoder: nn.Module
        Pre-trained encoder with a single-channel stem Conv3d layer.
    n_modalities: int
        Target number of input channels (must be ≥ 1).

    Returns
    -------
    encoder: nn.Module
        The same encoder object with its stem replaced (modified in-place).

    Raises
    ------
    ValueError
        If no Conv3d layer is found in the encoder.
    AssertionError
        If the stem Conv3d has more than 1 input channel.
    """
    if n_modalities == 1:
        return encoder

    # Find the first Conv3d and its parent module + attribute name
    stem_parent = None
    stem_attr = None
    stem_conv = None
    for module_name, module in encoder.named_modules():
        if isinstance(module, nn.Conv3d):
            # Navigate to the parent module
            parts = module_name.split(".")
            parent = encoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            stem_parent = parent
            stem_attr = parts[-1]
            stem_conv = module
            break

    if stem_conv is None:
        raise ValueError("No Conv3d layer found in encoder; cannot adapt to n_modalities.")

    assert stem_conv.in_channels == 1, (
        f"Stem weight repetition only supported when in_channels == 1, "
        f"got in_channels == {stem_conv.in_channels}."
    )

    # Build the new stem with repeated weights
    new_stem = nn.Conv3d(
        in_channels=n_modalities,
        out_channels=stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        dilation=stem_conv.dilation,
        groups=stem_conv.groups,
        bias=stem_conv.bias is not None,
        padding_mode=stem_conv.padding_mode,
    )

    with torch.no_grad():
        # Repeat along in_channels dim and normalise
        new_stem.weight.copy_(
            stem_conv.weight.repeat(1, n_modalities, 1, 1, 1) / n_modalities
        )
        if stem_conv.bias is not None:
            new_stem.bias.copy_(stem_conv.bias)

    _set_stem_conv(stem_parent, stem_attr, new_stem)
    return encoder
