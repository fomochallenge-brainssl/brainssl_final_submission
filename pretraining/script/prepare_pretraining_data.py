#!/usr/bin/env python3
"""
Merge the `dataset` and `participant_id` columns of a participants.tsv into a
single, globally-unique `participant_id` of the form `<dataset>_<participant_id>`.
This solves the eventual collision between subjects that have the same participant_id
but come from different datasets.

Usage
-----
    python merge_dataset_id.py --input participants.tsv --output participants_merged.tsv
"""

import argparse
import sys

import pandas as pd


def merge_dataset_into_id(
    input_path: str,
    output_path: str,
    suffix: str = '',
) -> None:
    sep = "\t"
    dataset_col = "dataset"
    id_col = "participant_id"
    ses_col = "session_id"

    df = pd.read_csv(input_path, sep=sep, dtype={dataset_col: str, id_col: str, ses_col: str})

    for col in (dataset_col, id_col, ses_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {input_path}. "
                f"Columns present: {list(df.columns)}"
            )

    original_rows_count = len(df)
    # compute global (dataset + sub-set) dataset id
    df[dataset_col] = df[dataset_col].apply(lambda s: compute_composed_id(s, suffix))

    # Saniteze subject id
    df[id_col] = df[id_col].apply(sanitize_subject_id)

    # Sanitize session id
    df[ses_col] = df[ses_col].apply(sanitize_session_id)
    
    # Compute global participant id
    df[id_col] = df[dataset_col] + "_" + df[id_col] + "_" + df[ses_col]

    # Remove the now-redundant dataset column; everything else is left as-is.
    df = df.drop(columns=[dataset_col, ses_col])

    df.to_csv(output_path, sep=sep, index=False)
    print(f"Wrote {len(df)} rows, {len(df.columns)} columns -> {output_path}")

    modified_rows_count = len(df)

    assert original_rows_count == modified_rows_count

def compute_composed_id(original_dataset_id, suffix=''):
    dataset_name_components = original_dataset_id[2:].split('_')
    dataset_id = dataset_name_components[0]
    dataset_id = dataset_id+suffix

    subset_pos = original_dataset_id.find('/')
    if subset_pos > -1:
        dataset_id = f"{dataset_id}_{original_dataset_id[subset_pos + 3:]}"

    return dataset_id

def sanitize_subject_id(subj_id):
    return subj_id.replace("sub-", "")

def sanitize_session_id(session_id):
    return session_id.replace("ses-", "")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge dataset column into participant_id columns in a participants tsv file"
    )
    parser.add_argument("--input", default="participants.tsv", help="Input TSV path.")
    parser.add_argument(
            "--dataset-suffix",
            default="",
            help="Suffix to add to dataset id. eg i if isotropic.",
        )
    parser.add_argument(
        "--output",
        default="participants_merged.tsv",
        help="Output TSV path (defaults to a new file so the original is preserved).",
    )
    args = parser.parse_args()

    merge_dataset_into_id(
        args.input,
        args.output,
        args.dataset_suffix,
    )


if __name__ == "__main__":
    main()
