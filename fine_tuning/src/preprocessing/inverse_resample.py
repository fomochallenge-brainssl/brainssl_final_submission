"""Inverse resampling: map a volume from isotropic inference space back onto
the exact native voxel grid of the original image. It inverses the
`nidl.Resample` transform when original affine and shape are known.

Background for inverse resampling implementation
------------------------------------------------
A medical image is not just a voxel array; it also has *geometry*: where each
voxel sits in physical (mm) space. That geometry is fully described by an
"affine" matrix (origin + spacing + orientation). "Resampling" means: given a
source image and a *target geometry*, compute the source's intensity at each
target voxel's physical location (via interpolation).

Two ways to specify the target geometry:
  1. by desired *spacing* only -- the tool then derives the output size/origin.
     This is what the forward step (``nidl.Resample``) does.
  2. by an *explicit, known grid* -- exact origin, spacing, orientation, size.

Inverting step 1 with another step-1 call does NOT reliably reproduce the
original grid: size is ceil-rounded and the origin follows a half-voxel
convention, so you can land one voxel off -- which would misalign the
prediction against the ground-truth label. So here we use approach 2: we
rebuild the *exact* original grid from the affine + shape we already know, and
resample onto it. Exactness is the whole point of this function.

We reuse three small helpers from ``nidl.Resample`` so our array<->image
conventions match the forward path exactly:
  - ``as_sitk(array, affine)``   : numpy array + affine  -> sitk image
  - ``from_sitk(image, dim)``    : sitk image            -> numpy array
  - ``_parse_interpolation(name)``: "linear"/"nearest"/... -> sitk constant
"""
from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch
from nidl.volume.transforms.preprocessing.spatial import Resample


def _build_reference_grid(affine: np.ndarray, shape: Tuple[int, int, int],
                          n_channels: int, dtype) -> sitk.Image:
    """Create an empty sitk image with the *exact* geometry we want the output
    to have: the original affine (origin/spacing/orientation) at the original
    shape. Only its geometry is used as the resampling target -- its (zero)
    contents are irrelevant.

    We get the geometry by making a tiny 1-voxel image with the right affine
    (``as_sitk`` fills in origin/spacing/orientation for us), then constructing
    a full-size image and copying that geometry over. This avoids allocating a
    full native-resolution array just to carry an affine.
    """
    stub_shape = (n_channels, 1, 1, 1) if n_channels else (1, 1, 1)
    stub = Resample.as_sitk(np.zeros(stub_shape, dtype=dtype), affine)

    # sitk sizes are (x, y, z). as_sitk transposes the array and GetImageFromArray
    # reverses axes again -- the two cancel -- so the sitk size equals the numpy
    # shape numerically. Pass `shape` as-is (do NOT reverse it).
    reference_grid = sitk.Image(tuple(int(s) for s in shape),
                                stub.GetPixelID(), stub.GetNumberOfComponentsPerPixel())
    reference_grid.SetOrigin(stub.GetOrigin())
    reference_grid.SetSpacing(stub.GetSpacing())
    reference_grid.SetDirection(stub.GetDirection())
    return reference_grid


def inverse_resample(
    volume: torch.Tensor,
    isotropic_affine: np.ndarray,
    original_affine: np.ndarray,
    original_shape: Tuple[int, int, int],
    interpolation: str = "linear",
) -> torch.Tensor:
    """Resample ``volume`` from isotropic space back onto the original grid.

    Parameters
    ----------
    volume: torch.Tensor of shape (C, H, W, D) or (H, W, D), in isotropic space.
        For a per-class logits canvas pass it as (num_classes, H, W, D) with
        ``interpolation="linear"`` and argmax *after* this call, in native
        space -- interpolating continuous logits is well-defined, interpolating
        discrete class indices is not.
    isotropic_affine: (4, 4) RAS affine of ``volume`` (the grid the forward
        ``Resample`` produced).
    original_affine: (4, 4) RAS affine of the native image to map back onto.
    original_shape: (H, W, D) spatial shape of the native image.
    interpolation: any name ``nidl.Resample`` accepts. Use "linear" for logits,
        "nearest" for an already-discrete label map.

    Returns
    -------
    torch.Tensor of shape (C, *original_shape) or (*original_shape), same
    dtype/device as the input.
    """
    # 1. torch -> numpy (sitk works on numpy/CPU). Remember dtype/device so we
    #    can hand back a tensor that matches the input.
    is_tensor = isinstance(volume, torch.Tensor)
    dtype = volume.dtype if is_tensor else None
    device = volume.device if is_tensor else None
    data = volume.detach().cpu().numpy() if is_tensor else np.asarray(volume)
    ndim = data.ndim  # 4 if channels present (C,H,W,D), else 3

    # 2. as_sitk assumes RAS-oriented affines; guard both, as the forward path does.
    original_affine = Resample._check_affine_ras(original_affine)
    isotropic_affine = Resample._check_affine_ras(isotropic_affine)

    # 3. Wrap the input as a sitk image `floating_sitk` carrying its (isotropic) geometry.
    floating_sitk = Resample.as_sitk(data, isotropic_affine)

    # 4. Build the exact target grid we want to land on (the native geometry).
    n_channels = data.shape[0] if ndim == 4 else 0
    reference_grid = _build_reference_grid(original_affine, original_shape, n_channels, data.dtype)

    # 5. Resample `floating_sitk` onto that target grid.
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(Resample._parse_interpolation(interpolation))
    resampler.SetReferenceImage(reference_grid)
    resampled = resampler.Execute(floating_sitk)

    # 6. sitk image -> numpy, and verify we landed on the intended shape.
    out = Resample.from_sitk(resampled, dim=ndim)
    assert out.shape[-3:] == tuple(original_shape), (
        f"inverse resample produced {out.shape[-3:]}, expected {tuple(original_shape)}"
    )

    # 7. numpy -> torch, if input is a tensor
    return torch.as_tensor(out, dtype=dtype, device=device) if is_tensor else out