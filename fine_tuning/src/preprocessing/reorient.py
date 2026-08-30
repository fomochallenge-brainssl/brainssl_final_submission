"""Reorient a volume (and its RAS affine) to canonical RAS+ axis order.

The orientation bookkeeping (which axis permutation/flip a RAS affine implies, and how
to invert it) reuses ``nibabel.orientations``
(https://nipy.org/nibabel/reference/nibabel.orientations.html), specifically
``io_orientation``, ``ornt_transform`` and ``inv_ornt_aff``.

``nibabel.orientations.apply_orientation`` -- the function that actually permutes/flips
the *array* -- is deliberately **not** reused here to avoid a torch->numpy conversion.
``_apply_ornt`` below is a torch-native reimplementation that reproduces its behaviour (flip on
the pre-transpose axis indices, then transpose by ``argsort(ornt[:, 0])``, restricted to a
tensor's trailing 3 (spatial) dims so any leading channel dim is left untouched, matching
how ``ForwardPreprocessing`` treats tensors as ``(C, H, W, D)`` / ``(H, W, D)``.
"""
from typing import Optional, Tuple

import numpy as np
import torch
from nibabel.orientations import io_orientation, inv_ornt_aff, ornt_transform

# The ornt of a volume that is already canonical RAS+: axis 0 -> +R, axis 1 -> +A,
# axis 2 -> +S, no flips. See ``nibabel.orientations.io_orientation``.
IDENTITY_ORNT = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.float64)


def _apply_ornt(tensor: torch.Tensor, ornt: np.ndarray) -> torch.Tensor:
    """Permute/flip the trailing 3 (spatial) dims of ``tensor`` per ``ornt``.

    Torch-native equivalent of ``nibabel.orientations.apply_orientation`` restricted to
    the last 3 dims -- any leading (channel) dims are passed through unpermuted.
    """
    ndim = tensor.dim()
    offset = ndim - 3
    # Flip first, on the pre-transpose (input) axis indices -- matches
    # apply_orientation's `for ax, flip in enumerate(ornt[:, 1])`.
    flip_dims = [offset + i for i in range(3) if ornt[i, 1] == -1]
    if flip_dims:
        tensor = torch.flip(tensor, dims=flip_dims)
    # Then transpose: apply_orientation uses `argsort(ornt[:, 0])`, not `ornt[:, 0]`
    # directly -- ornt[:, 0] is indexed by *input* axis (row i = "input axis i maps to
    # reference axis ornt[i, 0]"), so the permutation to hand to `.permute()` (indexed
    # by *output* axis) is its argsort.
    perm_spatial = np.argsort(ornt[:, 0])
    perm = list(range(offset)) + [offset + int(p) for p in perm_spatial]
    return tensor.permute(*perm).contiguous()


class RASReorient:
    """Reorient a volume to canonical RAS+ axis order, a no-op if already RAS+.

    Unlike ``Resample``/``Resize``, reorientation is a pure index permutation/flip --
    no interpolation is involved -- so a single instance reorients both image and label
    consistently in one call, rather than needing separate per-tensor instances.
    """

    def get_ornt(self, affine: np.ndarray) -> np.ndarray:
        """The ornt describing how ``affine``'s array axes relate to RAS+."""
        return io_orientation(np.asarray(affine, dtype=np.float64))

    def is_identity(self, ornt: np.ndarray) -> bool:
        """True if ``ornt`` describes a volume that is already RAS+ (no-op)."""
        return np.array_equal(ornt, IDENTITY_ORNT)

    def __call__(
        self,
        image: Optional[torch.Tensor],
        label: Optional[torch.Tensor],
        affine: np.ndarray,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], np.ndarray, np.ndarray]:
        """Reorient ``image``/``label`` (either may be ``None``) to RAS+.

        Parameters
        ----------
        image, label: torch.Tensor or None. Spatial dims are the trailing 3.
        affine: (4, 4) RAS affine describing ``image``'s/``label``'s current grid.

        Returns
        -------
        image, label: reoriented tensors (the same objects, unchanged, if already
            RAS+).
        new_affine: (4, 4) affine of the reoriented grid (``affine`` itself, unchanged,
            if already RAS+).
        ornt: the ornt that was applied (``IDENTITY_ORNT`` if already RAS+) -- callers
            need this to record/replay the inverse.
        """
        if image is None and label is None:
            raise ValueError('Nothing to reorient, ``image`` and ``label`` are both None')
        affine = np.asarray(affine, dtype=np.float64)
        ornt = self.get_ornt(affine)
        if self.is_identity(ornt):
            return image, label, affine, ornt

        shape = tuple(image.shape[-3:]) if image is not None else tuple(label.shape[-3:])
        if image is not None:
            image = _apply_ornt(image, ornt)
        if label is not None:
            label = _apply_ornt(label, ornt)
        new_affine = affine.dot(inv_ornt_aff(ornt, shape))
        return image, label, new_affine, ornt


def inverse_reorient(
    image: Optional[torch.Tensor],
    label: Optional[torch.Tensor],
    ornt: np.ndarray,
    affine: np.ndarray,
    shape: Tuple[int, int, int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], np.ndarray]:
    """Undo ``RASReorient``'s effect: reorient RAS+ ``image``/``label`` back to the
    native orientation described by the forward ``ornt``.

    Parameters
    ----------
    image, label: torch.Tensor or None, currently in RAS+ orientation.
    ornt: the ornt ``RASReorient`` applied going forward (native -> RAS+).
    affine: (4, 4) RAS+ affine of ``image``/``label`` as passed in.
    shape: spatial shape of ``image``/``label`` as passed in (the RAS+ shape).

    Returns
    -------
    image, label: reoriented back to native orientation.
    original_affine: (4, 4) affine of the native grid.
    """
    if np.array_equal(ornt, IDENTITY_ORNT):
        return image, label, np.asarray(affine, dtype=np.float64)

    inv_ornt = ornt_transform(IDENTITY_ORNT, ornt)
    if image is not None:
        image = _apply_ornt(image, inv_ornt)
    if label is not None:
        label = _apply_ornt(label, inv_ornt)
    original_affine = np.asarray(affine, dtype=np.float64).dot(inv_ornt_aff(inv_ornt, shape))
    return image, label, original_affine