from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from nidl.transforms import VolumeTransform
from nidl.volume.transforms.preprocessing.spatial.letterbox import Letterbox
from nidl.volume.transforms.preprocessing.spatial.resample import Resample
from nidl.volume.transforms.preprocessing.spatial.resize import Resize

from .reorient import RASReorient

_RESIZE_FN = {
    'resize': Resize,
    'letterbox': partial(
        Letterbox, padding_mode='constant', constant_values=0.0),
}
_INTERPOLATION_ITK = ['nearest', 'linear', 'bspline', 'cubic', 'gaussian', 'label_gaussian',
                      'hamming', 'cosine', 'welch', 'lanczos', 'blackman']

class ForwardPreprocessing:
    def __init__(
        self,
        resample_target_spacing: Optional[Union[float, int, Tuple[float, float, float], List[float]]] = None,
        resizing_method: Optional[str] = None,
        target_shape: Optional[Union[int, Tuple[int, int, int], List[float]]] = None,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest', # either 'linear' (float) or 'nearest' (int)
        normalization_transform: Optional[VolumeTransform] = None,
        ras_reorient: bool = True,
        keep_channels: Optional[Sequence[int]] = None,
    ):
        # Store necessary keys in metas to run the transforms
        # and their inverse
        self.meta_keys = ['original_affine']
        # Set defaults to False
        self.do_resampling = False
        self.do_resizing = False
        self.do_normalization = False

        self.keep_channels = None if keep_channels is None else list(keep_channels)
        self.do_reorientation = ras_reorient

        # Reorientation to canonical RAS+ axis order. If already RAS-aligned, does
        # nothing. On by default: downstream spatial steps (Resample) assume
        # RAS-aligned array axes.
        if self.do_reorientation:
            self.reorienter = RASReorient()

        # If spatial transform, set image and label interpolation modes
        if resample_target_spacing is not None or resizing_method is not None:
            self.image_interpolation = self._set_image_interpolation(image_interpolation)
            self.label_interpolation = self._set_label_interpolation(label_interpolation)
        if resizing_method is not None:
            self.target_shape = self._set_target_shape(target_shape)

        # Resampling
        if resample_target_spacing is not None:
            self.do_resampling = True
            self.resample_target_spacing = self._set_target_spacing(resample_target_spacing)
            self.img_resampler = Resample(target=self.resample_target_spacing,
                    interpolation=self.image_interpolation)
            self.label_resampler = Resample(target=self.resample_target_spacing,
                    interpolation=self.label_interpolation)

        # Resizing
        if resizing_method is not None:
            self.do_resizing = True
            if resizing_method in _RESIZE_FN:
                resizing_transform = _RESIZE_FN[resizing_method]
                self.img_resizer = resizing_transform(target_shape=self.target_shape,
                                       interpolation=self.image_interpolation)
                
                self.label_resizer = resizing_transform(target_shape=self.target_shape,
                                       interpolation=self.label_interpolation)
            else:
                raise ValueError(f'Valid resizing transforms are {_RESIZE_FN.keys()}; got {resizing_method}')

        # Intensity normalization
        if normalization_transform is not None:
            self.do_normalization = True
            if isinstance(normalization_transform, VolumeTransform):
                self.img_normalization = normalization_transform
            else:
                raise ValueError("Normalization transform should be a `nidl.transforms.VolumeTransform`; "
                                 f"got {normalization_transform} of type {type(normalization_transform)}.")

    def parse_data(
            self,
            image: Optional[torch.Tensor]=None,
            label: Optional[torch.Tensor]=None,
            metas: dict=None
        ):
        """ Checks correct input data."""
        if image is None and label is None:
            raise ValueError("At least one of `image` or `label` needs to be passed as input.")
        if metas is None:
            raise ValueError("Metadatas `meta` must be passed for preprocessing.")
        if not isinstance(metas, dict):
            raise TypeError(f'`meta` must be of type dict; got {metas} of type {type(metas)}.')

        # Check image and label are tensors with a valid spatial rank.
        # A label is allowed 1 extra case: 0/1d (classification/regression
        # target), which carries no spatial transform and is forwarded as-is.
        for name, tensor in (('image', image), ('label', label)):
            if tensor is None:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f'`{name}` must be a torch.Tensor; got {type(tensor)}.')
            is_scalar_label = name == 'label' and tensor.ndim <= 1
            if tensor.ndim not in (3, 4) and not is_scalar_label:
                raise ValueError(
                    f"`{name}` must be 3d or 4d (channel+spatial dimensions), "
                    f"got shape {tuple(tensor.shape)}"
                )

        # Check all necessary keys are in meta (e.g. `original_affine` if
        # resampling was configured -- see `self.meta_keys` in __init__).
        missing = [key for key in self.meta_keys if key not in metas]
        if missing:
            raise KeyError(
                f"metas is missing key(s) {missing}, required by the transforms "
                "this ForwardPreprocessing was configured with."
            )
 
        return image, label, metas
    
    def _apply_transform(self, image, label, metas):
        metas = dict(metas)  # never mutate the caller's dict
        spatial_label = self._is_spatial_label(label)
        inverse_ops = []  # ordered, self-describing -> replayed reversed to invert
 
        metas['original_shape'] = self._reference_shape(image, label)

        if self.keep_channels is not None and image is not None:
            image = self._select_channels(image)
        current_affine = np.asarray(metas['original_affine'], dtype=np.float64)

        # 0. Reorientation to canonical RAS+ axis order. Must run before resampling:
        # `Resample`/`inverse_resample` assume RAS-aligned array axes.
        if self.do_reorientation:
            pre_reorient_shape = self._reference_shape(image, label)
            reoriented_label = label if spatial_label else None
            image, reoriented_label, new_affine, ornt = self.reorienter(
                image, reoriented_label, current_affine)
            if spatial_label:
                label = reoriented_label
            if not self.reorienter.is_identity(ornt):
                inverse_ops.append(('reorient', {
                    'ornt': ornt,
                    'new_affine': new_affine,
                    'original_affine': current_affine,
                    'original_shape': pre_reorient_shape,
                }))
                current_affine = new_affine
 
        # 1. Resampling
        if self.do_resampling:
            affine = current_affine
            pre_shape = self._reference_shape(image, label)
            if image is not None:
                image = self.img_resampler(image, affine)
            if spatial_label:
                label = self.label_resampler(label, affine)
            new_affine = self._affine_after_resample(
                affine, self.resample_target_spacing)
            inverse_ops.append(('resample', {
                'new_affine': new_affine,
                'original_affine': affine,
                'original_shape': pre_shape,
            }))
            current_affine = new_affine
 
        # 2. Resizing (shape space; spacing is no longer meaningful past here).
        if self.do_resizing:
            pre_resize_shape = self._reference_shape(image, label)
            if image is not None:
                image = self.img_resizer(image)
            if spatial_label:
                label = self.label_resizer(label)
            inverse_ops.append(('resize', {
                'pre_resize_shape': pre_resize_shape,
                'transform_class': type(self.img_resizer).__name__,
                'target_shape': tuple(self.target_shape),
            }))
 
        # 3. Intensity normalization -- image only, geometry-agnostic, and
        #    deliberately NOT recorded in inverse_ops (inversion only rewinds
        #    spatial transforms, per spec).
        if self.do_normalization and image is not None:
            image = self.img_normalization(image)
 
        metas['new_shape'] = self._reference_shape(image, label)
        metas['new_affine'] = current_affine
        metas['inverse_ops'] = inverse_ops
        return image, label, metas

    def __call__(
        self,
        image: Optional[torch.Tensor]=None,
        label: Optional[torch.Tensor]=None,
        metas: dict=None
    ):
        image, label, metas = self.parse_data(image, label, metas)
        return self._apply_transform(image, label, metas)

    def _select_channels(self, image: torch.Tensor) -> torch.Tensor:
        """Keep `self.keep_channels` of a channel-first image, in that order."""
        num_channels = image.shape[0]
        out_of_range = [i for i in self.keep_channels
                        if i < 0 or i >= num_channels]
        if out_of_range:
            raise ValueError(
                f"`keep_channels`={self.keep_channels} refers to channel(s) "
                f"{out_of_range} of a {num_channels}-channel image. Indices are "
                "positions in the `.pt` file, so they must be < "
                "`task.dataset.n_modalities`."
            )
        return image[self.keep_channels]

    def _is_spatial_label(self, label: Optional[torch.Tensor]) -> bool:
        """True for a segmentation mask/logits (>=3 spatial dims); False for a
        classification/regression target (0/1d), which every spatial stage
        below leaves untouched."""
        return isinstance(label, torch.Tensor) and label.ndim >= 3

    def _reference_shape(self, image, label) -> Tuple[int, int, int]:
        """Spatial shape, read from whichever of image/label is
        present (image preferred)."""
        tensor = image if image is not None else label
        return tuple(int(s) for s in tensor.shape[-3:])
    
    def _set_image_interpolation(
        self,
        image_interpolation=None,
    ):
        if image_interpolation in _INTERPOLATION_ITK:
            return image_interpolation
        else:
            raise ValueError(f"Valid image interpolation mode are {_INTERPOLATION_ITK}; got {image_interpolation}")
        
    def _set_label_interpolation(
        self,
        label_interpolation=None,
    ):
        if label_interpolation in ['linear', 'nearest']:
            return label_interpolation
        else:
            raise ValueError(f"Valid label interpolation mode are 'nearest' for integer label maps and"
                              f"'linear' for logits maps; got {label_interpolation}")

    def _set_target_shape(
        self,
        target_shape=None,
    ):
        if target_shape is None:
            raise ValueError('`target_shape` is required for resizing methods.')
        elif isinstance(target_shape, int):
            return (target_shape, target_shape, target_shape)
        elif isinstance(target_shape, Sequence) and len(target_shape) == 3:
            return target_shape
        else:
            raise TypeError('`target_shape` should be int or Tuple[int, int, int]; '
                            f'got {target_shape} of type {type(target_shape)}.')
        
    def _set_target_spacing(
        self,
        target_spacing=None,
    ):
        if target_spacing is None:
            raise ValueError('`resample_target_spacing` is required for resampling.')
        elif isinstance(target_spacing, (int, float)):
            s = float(target_spacing)
            return (s, s, s)
        elif isinstance(target_spacing, Sequence) and len(target_spacing) == 3:
            return tuple(float(s) for s in target_spacing)
        else:
            raise TypeError('`resample_target_spacing` should be a number or '
                            f'Tuple[float, float, float]; got {target_spacing} '
                            f'of type {type(target_spacing)}.')

    def _affine_after_resample(
        self,
        source_affine: np.ndarray,
        target_spacing: Tuple[float, float, float]
    ) -> np.ndarray:
        """RAS affine of the grid produced by resampling to ``target_spacing``.
    
        ``nidl.Resample`` returns only the array, not the grid affine, so this
        rebuilds it: same orientation as the source, spacing replaced, origin
        shifted per ITK's half-voxel convention so the two grids stay physically
        co-registered (matches ``Resample.get_reference_image``).
        """
        src = np.asarray(source_affine, dtype=np.float64)
        old_spacing = np.linalg.norm(src[:3, :3], axis=0)
        rotation = src[:3, :3] / old_spacing
        new_spacing = np.asarray(target_spacing, dtype=np.float64)
    
        new_affine = np.eye(4)
        new_affine[:3, :3] = rotation * new_spacing
        origin_shift_index = 0.5 * (new_spacing / old_spacing - 1.0)
        new_affine[:3, 3] = src[:3, 3] + rotation @ (old_spacing * origin_shift_index)
        return new_affine

    def does_resizing(self):
        '''Whether this pipeline implements a resizing steps.'''
        return self.do_resizing