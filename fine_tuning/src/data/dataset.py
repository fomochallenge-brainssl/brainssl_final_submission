import json
import os
import pickle
import warnings
import re
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from copy import deepcopy

from ..preprocessing import ForwardPreprocessing


def _load_segmentation_from_nifti_fallback(pt_path: str) -> torch.Tensor:
    """Rebuild a stacked segmentation tensor when a ``.pt`` file is corrupted.

    Expects FOMO finetuning layout, e.g.::

        .../processed_data/SEG010.../Task_4/preprocessed/sub-34/ses-01/t2w.pt
        -> .../raw/Task_4/Task_4/preprocessed/sub-34/ses-01/t2w.nii.gz
        -> .../raw/Task_4/Task_4/labels/sub-34/ses-01/seg.nii.gz

    Rebuilt tensors are cached under ``~/.cache/brainssl/pt_fallback/`` so each
    corrupted sample is only rebuilt once.
    """
    import hashlib
    import nibabel as nib

    cache_dir = os.path.expanduser("~/.cache/brainssl/pt_fallback")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.sha256(pt_path.encode()).hexdigest() + ".pt"
    cache_path = os.path.join(cache_dir, cache_key)
    if os.path.isfile(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    match = re.search(
        r"/processed_data/[^/]+/(Task_\d+)/preprocessed/(sub-\d+)/(ses-\d+)/t2w\.pt$",
        pt_path,
    )
    if match is None:
        raise ValueError(f"Cannot infer raw NIfTI paths from {pt_path!r}.")

    task_dir, subject, session = match.groups()
    fomo_root = pt_path.split("/processed_data/")[0]
    raw_root = os.path.join(fomo_root, "raw", task_dir, task_dir)
    image_path = os.path.join(raw_root, "preprocessed", subject, session, "t2w.nii.gz")
    label_path = os.path.join(raw_root, "labels", subject, session, "seg.nii.gz")

    image = nib.load(image_path).get_fdata().astype(np.float32)
    label = nib.load(label_path).get_fdata().astype(np.float32)
    tensor = torch.from_numpy(np.stack([image, label], axis=0))
    torch.save(tensor, cache_path)
    return tensor


def _load_pt_tensor(path: str) -> torch.Tensor | list:
    try:
        with open(path, "rb") as img_file:
            return torch.load(img_file, map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        if path.endswith("/t2w.pt") or path.endswith("\\t2w.pt"):
            return _load_segmentation_from_nifti_fallback(path)
        raise exc

def _load_meta_dict(path: str) -> dict:
    """Load the ``.pkl`` metadata sidecar for a ``.pt`` sample.

    Every ``.pt`` sample must have a companion ``.pkl`` file at the same path
    (extension swapped) describing the image/label as stored on disk. Only
    ``original_affine`` is extracted here.

    Parameters
    ----------
    path:
        Path to the ``.pt`` file, as passed to ``fomo26_loader``.

    Returns
    -------
    dict
        ``{'original_affine': np.ndarray}``, the RAS affine (4, 4) of the
        image as stored on disk.

    Raises
    ------
    FileNotFoundError
        If the ``.pkl`` sidecar does not exist. A sample without a sidecar is
        treated as a data-integrity error.
    KeyError
        If the sidecar exists but does not carry ``d['nifti_metadata']['affine']``.
    """
    if not path.endswith('.pt'):
        raise ValueError(f"Expected a `.pt` path, got {path!r}.")
    meta_path = path.replace('.pt', '.pkl')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Missing metadata sidecar for {path!r}: expected {meta_path!r}. "
            "Every `.pt` sample must have a companion `.pkl` file."
        )
    with open(meta_path, "rb") as meta_file:
        d = pickle.load(meta_file)
    try:
        affine = d['nifti_metadata']['affine']
    except (KeyError, TypeError) as exc:
        raise KeyError(
            f"{meta_path!r} is missing d['nifti_metadata']['affine']."
        ) from exc
    return {'original_affine': np.asarray(affine, dtype=np.float64)}


class FOMO26Loader:
    def __init__(self, n_modalities: int):
        """Defines a loader that reads a ``.pt`` file and returns ``(image, label, meta)``.
        
        The ``.pt`` file is expected to store either:
    
        - A ``list`` of ``[image_tensor, label_tensor]`` for classification and
            regression tasks.
        - A stacked tensor ``(C+1, H, W, D)`` for segmentation tasks, where the
            last channel is the segmentation mask.
    
        ``meta`` comes from the ``.pt`` file's ``.pkl`` sidecar (see
        :func:`_load_meta_dict`) and carries at least ``original_affine``.
    
        Parameters
        ----------
        n_modalities:
            Expected number of image channels (modalities).  A ``ValueError`` is
            raised if the loaded image has a different number of channels.
        """
        self.n_modalities = n_modalities

    def __call__(self, path:str):
        if not path.endswith('.pt'):
            raise ValueError(f"Unsupported file type: {path}")
        meta = _load_meta_dict(path)
        data = _load_pt_tensor(path)
        if isinstance(data, list):
            image, label = data[0], data[1].squeeze().long()
        else:
            # segmentation: stacked (C+1, H, W, D), last channel is the mask
            image, label = data[:-1], data[-1:]
        if len(image) != self.n_modalities:
            raise ValueError(
                f"Expected {self.n_modalities} image channel(s) but found "
                f"{len(image)} in {path!r}."
            )
        return image, label, meta

def filter_df(df, cols, keys):
    '''Preserves order of `keys`. in new df.
    Resets index in new df'''
    # Filter df on cols keys
    if isinstance(cols, str):
        cols = [cols]
    # Check cols identify unique rows
    df_unique = df.drop_duplicates(subset=cols)
    if len(df_unique) != len(df):
        raise ValueError('Columns on which to perform split for folds do not uniquely identify ',
                        f'samples. Columns are {cols}')
    else:
        keys_df = pd.DataFrame(keys, columns=cols)
        # We use left=keys_df to preserve the order of input keys rather than original df.
        filtered_df = pd.merge(left=keys_df, right=df, on=cols, how='inner')
    return filtered_df

def keys_to_index(df: pd.DataFrame,
                  cols: list[str] | str,
                  keys: list[list[str]] | list[str]):
    '''For a list of (list of) `keys` identifying samples in `df` with respect to columns `cols` 
    returns the index in `df` of these samples.
    Cannot use directly the index in filter_df(df, cols, keys) as it is reinitialized.'''
    # Add a column with original index
    df_with_index = df.reset_index(drop=False, names='original_index')
    df_with_index_filtered = filter_df(df_with_index, cols, keys)
    return df_with_index_filtered['original_index'].to_list()


class FOMO26Dataset(Dataset):
    """Dataset for FOMO26 challenge tasks with multi-fold cross-validation support.

    Each task may have a different number of input modalities (T1, T2*, FLAIR,
    …).  Fold-based train/val/test sub-datasets are obtained via
    :meth:`get_split`.

    Parameters
    ----------
    rootdir:
        Root directory of the dataset.  Relative ``image_path`` values in
        ``df`` are joined with this prefix.  Pass ``""`` if paths in ``df``
        are already absolute.
    df:
        Participant metadata.  Accepted forms:

        - ``str`` — path to a TSV file (absolute or relative to ``rootdir``).
        - ``pd.Series`` — converted to a single-column DataFrame named
          ``image_col``.
        - ``pd.DataFrame`` — used directly.
    n_modalities:
        Number of imaging modalities expected per sample.
    num_classes:
        Number of output classes (classification / segmentation) or ``1`` for
        regression.
    task_type:
        One of ``'classification'``, ``'regression'``, ``'segmentation'``.
    folds:
        Cross-validation fold definitions.  Accepted forms:

        - ``list[dict]`` — each element maps split names (``'train'``,
          ``'val'``, ``'test'``) to lists of participant IDs.
        - ``str`` — path to a JSON file containing the list above (absolute
          or relative to ``rootdir``).
        - ``None`` — no folds; :meth:`get_split` will raise ``ValueError``.
    folds_cols:
        Column name(s) in ``df`` used to match fold participant IDs.  Required
        when ``folds`` is provided.
    image_col:
        Name of the column in ``df`` containing image file paths.
    preprocessing_transform:
        A :class:`~src.preprocessing.ForwardPreprocessing` instance, applied
        to ``(image, label, metas)`` right after loading (``metas`` comes
        from the ``.pt`` file's ``.pkl`` sidecar -- see
        :func:`get_fomo26_loader`). Resamples/resizes/normalizes on the fly
        and records the ops needed to invert predictions back to the native
        grid (``inverse_preprocessing``).
    augmentation_transform:
        Optional callable ``(image, label) -> (image, label)``, applied
        AFTER ``preprocessing_transform`. Intended for random data augmentation
        that must see and modify image and label together (see
        ``probing.data_augmentation.compose.default_segmentation_augmentation``).
    use_augmentation_on_val:    
        Whether to use data augmentations on the validation set
    return_raw_labels:
        When getting item returns image, label, raw_label. Where raw_label has no transform applied.
    name:
        Human-readable dataset name, used in ``__str__`` and sub-dataset
        names produced by :meth:`get_split`.
    """

    def __init__(
        self,
        rootdir: str,
        df: Union[pd.DataFrame, pd.Series, str],
        n_modalities: int,
        num_classes: int,
        task_type: str,
        folds: Optional[Union[list, str]] = None,
        folds_cols: Optional[Union[str, list]] = None,
        image_col: str = "image_path",
        trainval_transform: Optional[ForwardPreprocessing] = None,
        test_transform: Optional[ForwardPreprocessing] = None,
        augmentation_transform: Optional[Callable] = None,
        use_augmentation_on_val: Optional[bool] = False,
        target_transform: Optional[Callable] = None,
        return_raw_labels: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        super().__init__()

        # --- Load participants dataframe ---
        if isinstance(df, str):
            if not os.path.isabs(df):
                df = os.path.join(rootdir, df)
            df = pd.read_csv(df, sep='\t')
        elif isinstance(df, pd.Series):
            df = df.to_frame(name=image_col)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a DataFrame, Series, or path to a TSV file.")
        if image_col not in df.columns:
            raise ValueError(f"`{image_col}` column not found in DataFrame.")

        self.rootdir = rootdir
        self.image_col = image_col
        self.df = df.copy().reset_index(drop=True)  # Because some methods rely on index
        self.df[image_col] = self.df[image_col].apply(
            lambda rpath: rpath if os.path.isabs(rpath) else os.path.join(rootdir, rpath)
        )

        # --- Load folds ---
        if isinstance(folds, str):
            if not os.path.isabs(folds):
                folds = os.path.join(rootdir, folds)
            with open(folds, 'rb') as folds_file:
                folds = json.load(folds_file)
        if folds is None:
            self.folds = None
        elif isinstance(folds, list):
            self.folds = folds
        else:
            raise TypeError(
                "`folds` must be a list of dicts, a path to a JSON file, or None; "
                f"got {type(folds)}."
            )
        self.folds_cols = self._check_folds_cols(folds_cols)

        self.n_modalities = n_modalities
        self.num_classes = num_classes
        self.task_type = task_type
        self.trainval_transform = trainval_transform
        self.test_transform = test_transform
        self.augmentation_transform = augmentation_transform
        self.use_augmentation_on_val = use_augmentation_on_val
        self.target_transform = target_transform
        self.image_loader = FOMO26Loader(self.n_modalities)
        self.imgs = self.df[image_col].tolist()
        self.name = name
        self.return_raw_labels = return_raw_labels if return_raw_labels is not None else False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_folds_cols(self, folds_cols: Union[str, list]) -> Optional[Union[str, list]]:
        """Validate ``folds_cols`` against the DataFrame columns.

        Returns the validated value, or ``None`` when ``folds`` is ``None``.
        """
        if self.folds is None:
            return None
        if folds_cols is None:
            raise ValueError("`folds_cols` is required when `folds` is provided.")
        if isinstance(folds_cols, str):
            if folds_cols not in self.df.columns:
                raise ValueError(f"Column '{folds_cols}' not found in DataFrame.")
        elif isinstance(folds_cols, list):
            missing = set(folds_cols) - set(self.df.columns)
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame.")
        else:
            raise TypeError(
                f"`folds_cols` must be a str or list of str, got {type(folds_cols)}."
            )
        return folds_cols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_split(self, fold: int, split: str,
                  return_raw_labels: Optional[bool] = None) -> "FOMO26Dataset":
        """Return a new dataset containing only the samples for one fold/split.

        Parameters
        ----------
        fold:
            Zero-based fold index.  Must be less than ``len(self.folds)``.
        split:
            Split name, typically ``'train'``, ``'val'``, or ``'test'``.

        Returns
        -------
        FOMO26Dataset
            A dataset restricted to the requested subset. ``augmentation_transform``
            can be removed using the ``no_data_aug`` flag. Folds are not propagated,
            so calling :meth:`get_split` on the returned dataset will raise ValueError.
        """
        if self.folds is None:
            raise ValueError(
                "No folds defined — pass `folds` and `folds_cols` at construction time."
            )
        if fold >= len(self.folds):
            raise ValueError(
                f"Fold {fold} does not exist; only {len(self.folds)} fold(s) available."
            )
        if split not in self.folds[fold]:
            raise ValueError(
                f"Split '{split}' does not exist in fold {fold}. "
                f"Available splits: {list(self.folds[fold].keys())}."
            )

        split_keys = self.folds[fold][split]

        # Filter df on fold_cols keys. /!\ Resets index
        split_df = filter_df(self.df, self.folds_cols, split_keys)

        if split == 'train':
            trainval_transform = self.trainval_transform
            test_transform = None
            augmentation_transform = self.augmentation_transform
        elif split == 'val':
            trainval_transform = self.trainval_transform
            test_transform = None
            if self.use_augmentation_on_val:
                augmentation_transform = self.augmentation_transform
            else:
                augmentation_transform = None
        elif split == 'test':
            trainval_transform = None
            test_transform = self.test_transform
            augmentation_transform = None
        else:
            raise ValueError(f"Split currently supported are 'train', 'val' and 'test', got {split}.")
            
        if return_raw_labels is None:
            return_raw_labels = self.return_raw_labels

        return FOMO26Dataset(
            rootdir=self.rootdir,
            df=split_df,
            n_modalities=self.n_modalities,
            num_classes=self.num_classes,
            task_type=self.task_type,
            folds=None,
            folds_cols=None,
            image_col=self.image_col,
            trainval_transform=trainval_transform,
            test_transform=test_transform,
            augmentation_transform=augmentation_transform,
            target_transform=self.target_transform,
            return_raw_labels=return_raw_labels,
            name=f"{self.name}_{split}" if self.name else None,
        )

    def get_folds_indices(self) -> Iterable:
        '''Returns all folds and splits as indices. Merges train and
        val indices to yield only a train/test split. Compatible with
        sklearn `cv` parameter.
        '''

        if self.folds is None:
            raise ValueError(
                "No folds defined — pass `folds` and `folds_cols` at construction time."
            )

        cv = []
        for fold in self.folds:
            train_indices = keys_to_index(self.df, self.folds_cols, fold['train'])  # May contain the same key several times
            trainval_indices = train_indices
            if 'val' in fold:
                val_indices = keys_to_index(self.df, self.folds_cols, fold['val'])  # May contain the same key several times
                trainval_indices += val_indices
            test_indices = keys_to_index(self.df, self.folds_cols, fold['test'])
            cv.append([trainval_indices, test_indices])
        return cv

    def has_duplicates(self):
        # Check cols identify unique rows
        return bool(self.df.duplicated(subset=self.folds_cols).any())

    def remove_data_augmentation(self) -> None:
        '''Removes data augmentation transforms. Useful for validation and test sets.'''
        # Remove data augmentation transform.
        self.augmentation_transform = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image, labels, loader_meta = self.image_loader(self.imgs[idx])

        # Meta information on batch
        meta = {}

        if self.return_raw_labels:
            # Record raw labels to evaluate final segmentation in input space.
            meta['raw_labels'] = deepcopy(labels)

        # Joint image + label preprocessing transform 
        # with history tracking for inversion
        if self.trainval_transform is not None:
            if self.test_transform is not None:
                raise AttributeError('Trying to load input volumes and apply both `trainval_transform` and' \
                '`test_transform`. You probably want to apply only one of both. Set one of the two to `None`.')

            # Current pipeline: resample/resize/normalize on the fly, using
            # the affine read from the .pt file's .pkl sidecar (loader_meta).
            image, labels, transform_meta = self.trainval_transform(
                image=image, label=labels, metas=loader_meta,
            )
            if self.task_type == "segmentation":
                # Merge transform_meta to keep transform history needed to
                # invert predictions back to the native grid at eval time
                # -- see src.preprocessing.inverse_preprocessing.
                meta = meta | transform_meta
        
        if self.test_transform is not None:
            image, labels, transform_meta = self.test_transform(
                image=image, label=labels, metas=loader_meta,
            )
            if self.task_type == "segmentation":
                meta = meta | transform_meta


        # Random joint (image+label) augmentation
        if self.augmentation_transform is not None:
            # If classification or regression, only apply to image
            if self.task_type in ['regression', 'classification']:
                image, _ = self.augmentation_transform(image, None)
            elif self.task_type == 'segmentation':
                # If segmentation, apply to image and labels
                image, labels = self.augmentation_transform(image, labels)
        
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        if len(meta) > 0:
            return image, labels, meta
        return image, labels

    def __str__(self) -> str:
        base = super().__str__()
        return f"Dataset {self.name}: {base}" if self.name else base