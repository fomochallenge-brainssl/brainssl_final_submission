import nidl
import torch
from nidl.datasets.base import BaseDataset
import os
import glob
import warnings
import pandas as pd
import hashlib
import json
import fnmatch
from collections import defaultdict
import random

class PretrainingDataset(BaseDataset):
    def __init__(self, root, patterns, channels, subject_in_patterns, dataset_in_patterns,
                 split="train", targets=None, target_mapping=None,
                 transforms=None, mask=None, withdraw_subjects=None,
                 multimodal=False, n_views=2):

        super().__init__(
            root, patterns, channels, split=split, targets=targets,
            target_mapping=target_mapping, transforms=transforms, mask=mask,
            withdraw_subjects=withdraw_subjects)

        # CoMM multimodal mode: return `n_views` augmented views per modality
        self.multimodal = multimodal
        self.n_views = n_views

        # Extend to all subjects
        if not isinstance(subject_in_patterns, (list, tuple)):
            subject_in_patterns = [subject_in_patterns] * len(patterns)
        assert len(patterns) == len(subject_in_patterns)

        if not isinstance(dataset_in_patterns, (list, tuple)):
            dataset_in_patterns = [dataset_in_patterns] * len(patterns)
        assert len(patterns) == len(dataset_in_patterns)

        self.subject_in_patterns = subject_in_patterns

        with open(os.path.join(root, f"split_99_01_00.json")) as f:
            self.paths = json.load(f)[0][split]
            self._data = {}
            for idx, pattern in enumerate(patterns):
                # subject's id position in file path
                _sidx = subject_in_patterns[idx]
                # dataset id position in file path
                _didx = dataset_in_patterns[idx]
                # index and group files by [dataset number]_[subject id number]_[session id number]
                # for each subject, we get all scans of its scans of a certain modality
                _files = defaultdict(list)
                _pats = [pattern] if isinstance(pattern, str) else list(pattern)
                for path in self.paths:
                    if any(fnmatch.fnmatch(path, p) for p in _pats):
                        _files[self.compute_file_key(path, _didx, _sidx, _sidx + 1)].append(path)
                # store file paths in dataframe, grouping by modality
                # format: 
                #    {
                #       "T1w": [[file path subject 1, ...], [file path subject 2, ...] ...]               
                #       "T2w": [[file path subject 1, ...], [] ...]                
                #       .....
                #    }
                # If one modality is missing for a subject, than it's represented as empty list
                self._data[f"{self.channels[idx]}"] = [_files[subject] for subject in self._df["participant_id"].unique()]
            
            # Convert to dataframe and signal missing modalities
            # rows are subjects, columns are modalities, cells values are lists of paths
            # (i, j) contains the list of scans of subject i for modality j
            self._data = pd.DataFrame.from_dict(self._data)
            _missing_data = self._data[self._data.isnull().any(axis=1)]
            if len(_missing_data) > 0:
                warnings.warn(f"Missing file data in {split}!", UserWarning,
                            stacklevel=2)
            
            # returns a numpy array, where for each subject we have one lists of its scan per modality
            # [
            #    [[scans of modality t1w], [scans of modality t2w], ...]]
            #    [[scans of modality t1w], [scans of modality t2w], ...]]
            #    ....
            # ]
            self._data = self._data.values

        if not self.multimodal:
            # Unimodal mode (one scan per sample)
            self.flatten_to_unimodal()
        else:
            # Multimodal mode for CoMM: keep subjects, but drop those with no scans in any
            # of the selected modalities
            self._data = [row for row in self._data
                          if any(len(scans) > 0 for scans in row)]

        print(f"Samples count: {self.__len__()}\n")

        
    # Derive file key from path
    def compute_file_key(self, file_path, dataset_idx, subject_idx, session_idx):
        path_components = file_path.split(os.sep)
        dataset_id = self.sanitize_dataset(path_components[dataset_idx], path_components[dataset_idx - 1])
        subject_id = self.sanitize_id(path_components[subject_idx], "sub-")
        session_id = self.sanitize_id(path_components[session_idx], "ses-")

        return f"{dataset_id}_{subject_id}_{session_id}"

    # extract number from id
    def sanitize_id(self, id, exp):
        return id.replace(exp, "")

    # Extract dataset and subset number from id
    def sanitize_dataset(self, dataset_name, prev_dataset_name):
        if dataset_name.startswith('ds0'):
            subdataset_name = dataset_name[2:]
            dataset_name = prev_dataset_name.split("_")[0][2:]
            dataset_id = f"{dataset_name}_{subdataset_name}"
        else:
            dataset_name = dataset_name.split("_")[0][2:]
            dataset_id = f"{dataset_name}"
        
        return dataset_id

    def get_checksum(self, path):
        """ Hashing file.
        """
        with open(path) as of:
            checksum = hashlib.sha1(of.read()).hexdigest()
        return checksum

    def get_data(self, idx):
        """ Proper data indexing.
        """
        return self._data[idx], (self._targets[idx]
                                 if self._targets is not None else None)

    # Flatten partecipants list from 
    # [
    #   [scan 1 path, scan 2 path, ..., scan k path]
    #    ....
    # ]
    # to
    # [
    #   [scan 1 path]
    #   [scan 2 path]
    #   ...
    #   [scan k path]
    #   ...
    #  ]
    #
    def flatten_to_unimodal(self):
        flattened_subjects_list = []

        # Loop through subjects
        for i in range(len(self._data)):
            # Loop through modalities
            for j in range(len(self._data[i])):
                # Loop through all scans from same modality
                for file_path in self._data[i][j]:
                    flattened_subjects_list.append([file_path])            
        
        self._data = flattened_subjects_list
    
    def __len__(self):
        return len(self._data)

    # load multimodal image for CoMM: return a dict with augmented present modalities, 
    # `mod_idx` recording which modality slot each entry belongs to. 
    # For each present modality choose one run randomly
    # and produce `n_views` augmented views of it.
    def _get_multimodal(self, idx):
        scans = self._data[idx]              # length M, each a list of scan paths
        mods = []                            # indices of present modalities
        views = [[] for _ in range(self.n_views)]
        for m, paths in enumerate(scans):
            if len(paths) == 0:              # skip absent modalities entirely
                continue
            vol = torch.load(random.choice(paths), map_location="cpu")
            # MultiViewsTransform returns a list of `n_views` augmented views.
            mv = self.transforms(vol) if self.transforms is not None else [vol] * self.n_views
            assert len(mv) == self.n_views, (
                f"Transform returned {len(mv)} views, expected n_views={self.n_views}. "
                "Use nidl.transforms.MultiViewsTransform with a matching n_views.")
            for v in range(self.n_views):
                views[v].append(mv[v])
            mods.append(m)

        out = {"mod_idx": torch.tensor(mods, dtype=torch.long)}
        for v in range(self.n_views):
            out[f"aug{v + 1}"] = torch.stack(views[v])   # (P_i, 1, H, W, D)
        return out

    # Load image corresponding to the required sample
    def __getitem__(self, idx):
        if self.multimodal:
            return self._get_multimodal(idx)

        subj_images = []
        output_tensor = None

        modalities_count = len(self._data[idx])

        if modalities_count > 1:
            # In the multimodal case, get all subject's scans
            for i in range(modalities_count):
                # loop through all images of modality
                for image_path in self._data[idx][i]:
                    # shape: (1, H, W, D)
                    if self.transforms is not None:
                        transformed_image = self.transforms(torch.load(image_path, map_location='cpu'))
                    else:
                        transformed_image = torch.load(image_path, map_location='cpu')

                    subj_images.append(transformed_image)

            output_tensor = torch.cat(subj_images, dim = 0)
        else:
            if self.transforms is not None:
                transformed_image = self.transforms(torch.load(self._data[idx][0], map_location='cpu'))
            else:
                transformed_image = torch.load(self._data[idx][0], map_location='cpu')

            output_tensor = transformed_image

        # shape: (C, H, W, D) where C = modalities count
        return output_tensor