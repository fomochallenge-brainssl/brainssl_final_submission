import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
import sys

# Ensure the repo root is on the path when calling from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import rootutils

rootutils.setup_root(__file__, indicator=".env")

# Define local paths
FOMO_ROOT = os.getenv('FOMO_FINETUNING_DATA_ROOT')
PROJECT_ROOT = os.getenv('PROJECT_ROOT')


RANDOM_STATE = 42
ONLY_CV = True
n_splits = 10
cv_type = 'mccv'
OVERSAMPLE_MINORITY = True  # only applies to classification + mccv; train/val only
split_suffix = '10foldsMCCV_20260724'
trainvaltest_sizes = {
    "CLS002_FOMO26_Infarct": (0.5,0.2,0.3),
    "CLS003_FOMO26_Polymicrogyria":(0.5,0.2,0.3),
    "REGR002_FOMO26_BrainAge": (0.7,0.15,0.15),
    "SEG009_FOMO26_Meningioma": (0.55,0.15,0.3),
    "SEG010_FOMO26_TrigeminalNeuralgia": (0.5,0.2,0.3),
    "SEG003i_ISLES22_ADCDWI_ISO": (0.2,0.3,0.5),
}

PATH_DATASETS = os.path.join(FOMO_ROOT, 'processed_data')
datasets = os.listdir(PATH_DATASETS)


def main():
    for dataset in datasets:
        print(f'Preparing dataset: {dataset}')
        path_dataset = os.path.join(PATH_DATASETS, dataset)
        task_type = get_task_type(dataset)
        paths_file = os.path.join(path_dataset, 'paths.json')
        with open(paths_file, 'rb') as f:
            paths = json.load(f)
        dataset_meta_file = os.path.join(path_dataset, 'dataset.json')
        with open(dataset_meta_file, 'rb') as f:
            dataset_meta = json.load(f)

        path_df = os.path.join(path_dataset, 'participants.tsv')

        if not ONLY_CV:
            data = []
            for path in paths:
                sub = {}
                sub['participant_id'], sub['session_id'] = extract_metadata_from_bids(path)
                sub['image_path'] = os.path.relpath(path, FOMO_ROOT)
                data.append(sub)

            # Create participants.tsv
            df = pd.DataFrame(data)
            df = df.sort_values(by='participant_id').reset_index(drop=True)
            df.to_csv(path_df, sep='\t', index=False)
        
        else:
            df = pd.read_csv(path_df, sep='\t')
        # --- Build CV folds ---
        # For classification, merge labels so we can stratify.
        # For regression/segmentation y is unused (KFold ignores it).
        indices = np.array(df.index)
        y = None

        train_ratio, val_ratio, test_ratio = trainvaltest_sizes[dataset]
        assert train_ratio + val_ratio + test_ratio == 1.0, (
            "Train, validation, and test ratios must sum to 1, but got {}, {}, {}".format(train_ratio, val_ratio, test_ratio)
        )

        if task_type == 'classification':
            labels_path = os.path.join(FOMO_ROOT, 'raw_labels', dataset, 'labels.csv')
            labels = pd.read_csv(labels_path, names=['image_path', 'label'])
            labels['image_path'] = labels['image_path'].apply(
                lambda p: os.path.relpath(p, FOMO_ROOT) + '.pt'
            )
            df_with_labels = pd.merge(df, labels, on=['image_path'], how='inner')
            if len(df_with_labels) != len(df) or len(df_with_labels) != len(labels):
                raise RuntimeError(
                    f'Length mismatch: merged={len(df_with_labels)}, '
                    f'df={len(df)}, labels={len(labels)}'
                )
            df = df_with_labels
            indices = np.array(df.index)
            y = df['label'].values
            if cv_type == 'mccv':
                splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=RANDOM_STATE)
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        else:
            if cv_type == 'mccv':
                splitter = ShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=RANDOM_STATE)
            else:
                splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        if cv_type == 'mccv':
            folds = []
            if task_type == 'classification':
                folds_inter = []
                for i in range(n_splits):
                    trainval_idx, test_idx = balanced_stratified_test_split(
                        indices, y, test_ratio, random_state=RANDOM_STATE + i
                    )
                    folds_inter.append((trainval_idx, test_idx))
            else:
                folds_inter = [fold_idx for fold_idx in splitter.split(indices, y)]

            for i, (trainval_idx, test_idx) in enumerate(folds_inter):
                if task_type == 'classification':
                    val_ratio_trainval = val_ratio / (train_ratio + val_ratio)
                    val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_trainval, random_state=RANDOM_STATE)
                    train_idx, val_idx = next(val_splitter.split(trainval_idx, y[trainval_idx]))
                    train_idx = trainval_idx[train_idx]
                    val_idx = trainval_idx[val_idx]

                    if OVERSAMPLE_MINORITY:
                        train_idx = oversample_to_balance(train_idx, y, random_state=RANDOM_STATE + i)
                        val_idx = oversample_to_balance(val_idx, y, random_state=RANDOM_STATE + i + 1000)
                else:
                    val_ratio_trainval = val_ratio / (train_ratio + val_ratio)
                    val_splitter = ShuffleSplit(n_splits=1, test_size=val_ratio_trainval, random_state=RANDOM_STATE)
                    train_idx, val_idx = next(val_splitter.split(trainval_idx, None))
                    train_idx = trainval_idx[train_idx]
                    val_idx = trainval_idx[val_idx]

                fold = {
                    'train': df.iloc[train_idx]['participant_id'].tolist(),
                    'val': df.iloc[val_idx]['participant_id'].tolist(),
                    'test': df.iloc[test_idx]['participant_id'].tolist(),
                }
                folds.append(fold)

        else:
            # Step 1: produce 7 base groups f0…f6.
            # KFold.split accepts y and silently ignores it, so this line works for all task types.
            base_groups = [test_idx for _, test_idx in splitter.split(indices, y)]

            # Step 2: rolling window assignment.
            # Fold k: train = f_k…f_{k+4}, val = f_{k+5 mod 7}, test = f_{k+6 mod 7}
            folds = []
            for k in range(n_splits):
                train_idx = np.concatenate([base_groups[(k + i) % n_splits] for i in range(5)])
                val_idx   = base_groups[(k + 5) % n_splits]
                test_idx  = base_groups[(k + 6) % n_splits]
                fold = {
                    'train': df.iloc[train_idx]['participant_id'].tolist(),
                    'val': df.iloc[val_idx]['participant_id'].tolist(),
                    'test': df.iloc[test_idx]['participant_id'].tolist(),
                }
                folds.append(fold)

        # Save folds
        path_folds = os.path.join(path_dataset, f'split_{split_suffix}.json')
        with open(path_folds, 'w') as f:
            json.dump(folds, f, indent=4)

        print(f'Sizes for one fold: train:{len(folds[0]['train'])} ,',
              f'val:{len(folds[0]['val'])} ,',
              f'test:{len(folds[0]['test'])}')

        # Write the YAML task config
        dataset_config = {
            '_target_': 'src.data.FOMO26Dataset',
            '_partial_': True,
            'rootdir': "${oc.env:FOMO_FINETUNING_DATA_ROOT}",
            'df': os.path.relpath(path_df, FOMO_ROOT),
            'n_modalities': dataset_meta['dataset_config']['n_modalities'],
            'num_classes': dataset_meta['dataset_config']['n_classes'],
            'folds': os.path.relpath(path_folds, FOMO_ROOT),
            'folds_cols': 'participant_id',
            'image_col': 'image_path',
            'task_type': task_type,
            'name': dataset_meta['dataset_config']['task_name'],
        }
        path_config = os.path.join(PROJECT_ROOT, 'configs', 'task', dataset + '.yaml')
        with open(path_config, 'w') as f:
            yaml.dump(dataset_config, f, indent=4, sort_keys=False)


def get_task_type(dataset_name):
    """Infer task type from the dataset name prefix (CLS / REG / SEG)."""
    if dataset_name.startswith('CLS'):
        return 'classification'
    elif dataset_name.startswith('REG'):
        return 'regression'
    elif dataset_name.startswith('SEG'):
        return 'segmentation'


def extract_metadata_from_bids(path):
    """Extract participant_id and session_id from a BIDS-formatted image path."""
    # Harmonize path format (CLS003 uses underscores instead of hyphens)
    path = path.replace('sub_', 'sub-').replace('ses_', 'ses-')
    pattern = re.compile(r'\/(sub-\w+)\/(ses-\d+)\/')
    match = pattern.search(path)
    if match:
        return match.groups()
    raise RuntimeError(f'Could not match path {path!r} with expected pattern {pattern.pattern!r}')

def balanced_stratified_test_split(indices, y, test_ratio, random_state):
    """Draw a test set with an exact equal number of samples per class.
    Returns (trainval_idx, test_idx)."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    n_test_per_class = int(np.ceil(min(counts) * test_ratio))
    if n_test_per_class < 1:
        raise RuntimeError(
            f'test_ratio={test_ratio} too small for smallest class '
            f'(count={min(counts)}) to yield a balanced test set.'
        )
    test_idx = []
    for c in classes:
        class_idx = indices[y == c]
        chosen = rng.choice(class_idx, size=n_test_per_class, replace=False)
        test_idx.append(chosen)
    test_idx = np.concatenate(test_idx)
    trainval_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
    rng.shuffle(trainval_idx)
    return trainval_idx, test_idx


def oversample_to_balance(idx, y, random_state):
    """Oversample minority classes (with replacement) to match the majority
    class count within idx. Returns a shuffled index array with duplicates."""
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y[idx], return_counts=True)
    max_count = counts.max()
    balanced_idx = []
    for c in classes:
        class_idx = idx[y[idx] == c]
        if len(class_idx) < max_count:
            extra = rng.choice(class_idx, size=max_count - len(class_idx), replace=True)
            class_idx = np.concatenate([class_idx, extra])
        balanced_idx.append(class_idx)
    balanced_idx = np.concatenate(balanced_idx)
    rng.shuffle(balanced_idx)
    return balanced_idx

if __name__ == "__main__":
    main()
