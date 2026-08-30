from typing import Optional, Tuple

import numpy as np
import torch

from nidl.volume.transforms.preprocessing.spatial.resize import Resize
from nidl.volume.transforms.preprocessing.spatial.inverse_letterbox import InverseLetterbox

from .forward import _INTERPOLATION_ITK
from .inverse_resample import inverse_resample
from .reorient import inverse_reorient

# Resize kinds ForwardPreprocessing can currently produce (see forward.py's
# _RESIZE_FN).
_SUPPORTED_RESIZE_KINDS = ('Resize', 'Letterbox')


def _parse_input(
    image: Optional[torch.Tensor],
    label: Optional[torch.Tensor],
    metas: dict,
    interpolation_img: str,
    interpolation_label: str,
):
    """Validate inputs, mirroring ForwardPreprocessing.parse_data."""
    if image is None and label is None:
        raise ValueError("At least one of `image` or `label` needs to be passed as input.")
    if not isinstance(metas, dict):
        raise TypeError(f'`metas` must be of type dict; got {metas} of type {type(metas)}.')

    for name, tensor in (('image', image), ('label', label)):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'`{name}` must be a torch.Tensor; got {type(tensor)}.')
        if tensor.ndim not in (3, 4):
            raise ValueError(
                f"`{name}` must be 3d or 4d (channel+spatial dimensions), "
                f"got shape {tuple(tensor.shape)}. Unlike the forward pass, "
                "the inverse does not accept a 0/1d classification/regression "
                "label -- there is no spatial transform to invert for one."
            )

    required_keys = ('inverse_ops', 'original_affine', 'original_shape')
    missing = [key for key in required_keys if key not in metas]
    if missing:
        raise KeyError(
            f"metas is missing key(s) {missing}. `metas` must be the dict "
            "returned by ForwardPreprocessing (or carry the same keys)."
        )

    if interpolation_img not in _INTERPOLATION_ITK:
        raise ValueError(f"Valid image interpolation modes are {_INTERPOLATION_ITK}; "
                         f"got {interpolation_img}")
    if interpolation_label not in ('linear', 'nearest'):
        raise ValueError("Valid label interpolation modes are 'nearest' (discrete label) "
                         f"or 'linear' (logits); got {interpolation_label}")

    return image, label, metas


def _invert_resize(image, label, op: dict, interpolation_img: str, interpolation_label: str):
    """Undo a single 'resize' stage, dispatching on which transform produced it."""
    pre_shape = tuple(op['pre_resize_shape'])
    kind = op['transform_class']
    if kind not in _SUPPORTED_RESIZE_KINDS:
        raise ValueError(f"Cannot invert resize of unsupported kind: {kind}")

    if kind == 'Resize':
        if image is not None:
            image = Resize(pre_shape, interpolation=interpolation_img)(image)
        if label is not None:
            label = Resize(pre_shape, interpolation=interpolation_label)(label)
    elif kind == 'Letterbox':
        # InverseLetterbox is constructed with the letterbox's own target_shape
        # and called with the shape to restore to (pre_shape).
        target_shape = op['target_shape']
        if image is not None:
            image = InverseLetterbox(target_shape, interpolation_img)(image, pre_shape)
        if label is not None:
            label = InverseLetterbox(target_shape, interpolation_label)(label, pre_shape)
    return image, label


def _invert_resample(image, label, op: dict, interpolation_img: str, interpolation_label: str):
    """Undo a single 'resample' stage via inverse_resample (exact-grid rewind)."""
    new_affine = op['new_affine']
    original_affine = op['original_affine']
    original_shape = tuple(op['original_shape'])

    if image is not None:
        image = inverse_resample(image, new_affine, original_affine, original_shape,
                                 interpolation=interpolation_img)
    if label is not None:
        label = inverse_resample(label, new_affine, original_affine, original_shape,
                                 interpolation=interpolation_label)
    return image, label

def _invert_reorient(image, label, op: dict):
    """Undo a single 'reorient' stage: reorient back to the pre-reorientation
    (native) axis order. Pure permutation/flip -- unlike resize/resample, there is
    no interpolation choice to make.
    """
    ornt = op['ornt']
    current_affine = op['new_affine']
    current_shape = tuple(image.shape[-3:]) if image is not None else tuple(label.shape[-3:])
    image, label, _ = inverse_reorient(image, label, ornt, current_affine, current_shape)
    return image, label

def inverse_preprocessing(
    image: Optional[torch.Tensor] = None,
    label: Optional[torch.Tensor] = None,
    metas: dict = None,
    interpolation_img: str = 'linear',
    interpolation_label: str = 'nearest',
):
    """Rewind ForwardPreprocessing's spatial transforms back to the native grid.

    Data-driven: replays ``metas['inverse_ops']`` (built by ForwardPreprocessing)
    in reverse order. Only the spatial stages (reorient, resample, resize) are inverted.

    Parameters
    ----------
    image, label: torch.Tensor or None. At least one must be given. If given,
        `label` must be 3d/4d (a spatial mask or per-class logits) -- a
        0/1d classification/regression label is rejected: it carries no
        spatial transform, so there is nothing to invert.
    metas: dict produced by ForwardPreprocessing (must carry `inverse_ops`,
        `original_affine`, `original_shape`).
    interpolation_img: any nidl-supported image interpolation mode. Used only
        if the forward pass resampled/resized the image.
    interpolation_label: 'linear' (for logits) or 'nearest' (for an
        already-discrete label).

    Returns
    -------
    image, label, metas, nifti_metadata: `image`/`label` are back on the
    native grid (None passed through as None). `metas` gains `current_affine`
    / `current_shape`, both equal to the original geometry (full rewind).
    `nifti_metadata` is a standalone dict with the minimum needed to save a
    NIfTI from the returned image/label: `{'affine': ..., 'original_spacing':
    ...}` (the latter only if present in the input `metas`).
    """
    image, label, metas = _parse_input(image, label, metas, interpolation_img, interpolation_label)

    for name, op in reversed(metas['inverse_ops']):
        if name == 'resize':
            image, label = _invert_resize(image, label, op, interpolation_img, interpolation_label)
        elif name == 'resample':
            image, label = _invert_resample(image, label, op, interpolation_img, interpolation_label)
        elif name == 'reorient':
            image, label = _invert_reorient(image, label, op)
        else:
            raise ValueError(f"Unknown inverse op recorded in metas: {name!r}")

    original_affine = np.asarray(metas['original_affine'], dtype=np.float64)
    original_shape = tuple(metas['original_shape'])

    # Sanity check: a full rewind must land exactly back on the original shape.
    for name, tensor in (('image', image), ('label', label)):
        if tensor is not None and tuple(tensor.shape[-3:]) != original_shape:
            raise RuntimeError(
                f"inverse_preprocessing produced {name} of shape "
                f"{tuple(tensor.shape[-3:])}, expected {original_shape}."
            )

    metas = dict(metas)
    metas['current_affine'] = original_affine
    metas['current_shape'] = original_shape

    nifti_metadata = {'affine': original_affine}
    if 'original_spacing' in metas:
        nifti_metadata['original_spacing'] = metas['original_spacing']

    return image, label, metas, nifti_metadata