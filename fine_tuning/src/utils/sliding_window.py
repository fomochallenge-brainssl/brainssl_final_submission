"""Sliding-window inference for segmentation models on native-resolution volumes.

Adapted from the yucca/gardening_tools/asparagus implementations. Used only at test time
for segmentation tasks with ``eval.sliding_window.enabled=true``; never used
during training or validation.

References
----------
https://github.com/Sllambias/asparagus/blob/main/asparagus/modules/lightning_modules/segmentation_module.py
https://github.com/Sllambias/gardening_tools/blob/main/gardening_tools/modules/networks/BaseNet.py
https://github.com/Sllambias/gardening_tools/blob/main/gardening_tools/modules/networks/utils.py

"""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_steps_for_sliding_window(shape: Tuple[int, int, int], patch_size: Tuple[int, int, int],
                                  overlap: float) -> list:
    """Compute per-axis patch start indices covering ``shape`` with ``patch_size``
    windows and the given ``overlap`` fraction.

    The last step per axis is always ``shape[i] - patch_size[i]`` so the patch
    grid exactly reaches the far edge of the volume. Requires
    ``shape[i] >= patch_size[i]`` for every axis (see ``pad_to_patch_size``).
    """
    assert 0 <= overlap < 1, "overlap must satisfy 0 <= overlap < 1"
    steplist = []
    for i, patch in enumerate(patch_size):
        stride = max(1, int(np.floor(patch * (1 - overlap))))
        final_step = shape[i] - patch
        assert final_step >= 0, (
            f"axis {i}: shape {shape[i]} smaller than patch_size {patch}; "
            "call pad_to_patch_size first"
        )
        steps = np.arange(0, shape[i], stride)
        steps = steps[steps < final_step]
        steps = np.append(steps, final_step)
        steplist.append(steps.astype(int))
    return steplist


def pad_to_patch_size(
    image: torch.Tensor, patch_size: Tuple[int, int, int]
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Zero-pad the trailing 3 spatial dims of ``image`` (..., H, W, D) up to at
    least ``patch_size``, so the sliding window always fits.

    Padding is appended at the end of each axis only, so inversion
    (``unpad_to_original``) is a simple trailing slice.

    Returns
    -------
    padded_image, pad_amounts: pad_amounts is (pad_h, pad_w, pad_d), the number
    of voxels appended along each axis (0 if no padding was needed).
    """
    spatial_shape = image.shape[-3:]
    pad_amounts = tuple(max(0, p - s) for s, p in zip(spatial_shape, patch_size))
    if not any(pad_amounts):
        return image, pad_amounts
    # F.pad takes (before, after) pairs per axis, innermost axis first.
    pad_spec = []
    for amt in reversed(pad_amounts):
        pad_spec.extend([0, amt])
    return F.pad(image, pad_spec, mode="constant", value=0.0), pad_amounts


def unpad_to_original(volume: torch.Tensor, pad_amounts: Tuple[int, int, int]) -> torch.Tensor:
    """Inverse of ``pad_to_patch_size``: removes the trailing padding added
    along each of the last 3 spatial dims.
    """
    if not any(pad_amounts):
        return volume
    slices = [slice(None)] * (volume.dim() - 3)
    for amt, size in zip(pad_amounts, volume.shape[-3:]):
        slices.append(slice(0, size - amt))
    return volume[tuple(slices)]


def sliding_window_predict(
    model, image: torch.Tensor, patch_size: Tuple[int, int, int],
    overlap: float, num_classes: int, sw_batch_size: int = 1,
) -> torch.Tensor:
    """Run ``model`` over overlapping patches of ``image`` and stitch the
    logits back into a full-resolution canvas.

    Parameters
    ----------
    model: callable, maps (1, C, ph, pw, pd) -> (1, num_classes, ph, pw, pd).
    image: torch.Tensor of shape (1, C, H, W, D), already padded to at least
        ``patch_size`` (see ``pad_to_patch_size``) and on ``model``'s device.
    patch_size: (ph, pw, pd), the spatial size the model was trained on.
    overlap: fraction in [0, 1) of patch overlap between consecutive steps.
    num_classes: number of output channels the decoder produces.
    sw_batch_size: number of patches forwarded through ``model`` at once.
    
    Returns
    -------
    logits: torch.Tensor of shape (1, num_classes, H, W, D).

    Notes
    -----
    Overlapping regions are summed, not averaged. Every class channel
    accumulates the same per-voxel count of contributing patches, so
    ``argmax`` over the class dimension is unaffected by this count -- summing
    and averaging give identical hard predictions. This only holds because the
    caller consumes hard (argmax) predictions; do not reuse this canvas as a
    calibrated probability/logit map without dividing by the coverage count.
    """
    assert image.dim() == 5 and image.shape[0] == 1, \
        f"expected (1, C, H, W, D), got {tuple(image.shape)}"
    px, py, pz = patch_size
    original_shape = tuple(image.shape[2:])

    # Pad
    image, pad_amounts = pad_to_patch_size(image, patch_size)

    padded_shape = tuple(image.shape[2:])
    canvas = torch.zeros((1, num_classes, *padded_shape), device=image.device, dtype=torch.float32)

    x_steps, y_steps, z_steps = get_steps_for_sliding_window(padded_shape, patch_size, overlap)
    coords = [(xs, ys, zs) for xs in x_steps for ys in y_steps for zs in z_steps]
    print(
        f"Sliding-window prediction on image of shape {original_shape}. "
        f"Prediction over {len(coords)} sub-volumes of shape {patch_size} "
        f"with batch size {sw_batch_size}."
    )
    # Batch-prediction of sub-volumes
    for i in range(0, len(coords), sw_batch_size):
        chunk = coords[i:i + sw_batch_size]
        patches = torch.cat(
            [image[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] for xs, ys, zs in chunk],
            dim=0,
        )  # (len(chunk), C, px, py, pz)
        out = model(patches)  # (len(chunk), num_classes, px, py, pz)
        for out_i, (xs, ys, zs) in zip(out, chunk):
            canvas[:, :, xs:xs + px, ys:ys + py, zs:zs + pz] += out_i.unsqueeze(0)

    # Unpad
    canvas = unpad_to_original(canvas, pad_amounts)
    return canvas