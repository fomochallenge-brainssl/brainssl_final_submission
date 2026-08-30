#!/usr/bin/env python3
"""
    Computes the .tsv participants list needed by pre-training dataset code

Usage
-----
    python split_pretraining_data.py --split-json [path to json] --output-dir [path to tsv files directory]
"""
import argparse
import json
import os
import re
import pandas as pd

def load_json(path):
    with open(path) as f:
        return json.load(f)


# Extracts desired split form split.json file
def select_split(split_data, split):
    if isinstance(split_data, dict):
        entry = split_data
    else:
        raise TypeError(
            f"Unexpected split.json top-level type {type(split_data).__name__}."
        )
    if split not in entry:
        raise KeyError(
            f"split '{split}' not in split.json entry; available: {list(entry.keys())}"
        )
    return entry[split]

# Extracts participant_id of the form <dataset>[_<subdataset>]_<subject>_<session> from file path
def path_to_participant_id(path):
    subject_re = re.compile(r"^sub-") 
    subdataset_re = re.compile(r"^ds")
    session_re = re.compile(r"^ses-")
    parts = path.split(os.sep)

    subj_idx = None
    for i, part in enumerate(parts):
        if subject_re.match(part):
            subj_idx = i
            break
    if subj_idx is None:
        raise ValueError(
            f"No subject component matching {subject_re.pattern!r} in path: {path}"
        )

    subject = sanitize_id(parts[subj_idx], "sub-") 
    if subj_idx == 0:
        raise ValueError(f"Subject has no parent dataset component in path: {path}")

    prev = parts[subj_idx - 1]
    if subdataset_re.match(prev):
        if subj_idx < 2:
            raise ValueError(
                f"Sub-dataset '{prev}' has no parent dataset component in path: {path}"
            )
        dataset = parts[subj_idx - 2]
        global_id = f"{sanitize_dataset(dataset)}_{prev[2:]}_{subject}"
    else:
        dataset = prev
        global_id = f"{sanitize_dataset(dataset)}_{subject}"

    session_idx = None
    for i, part in enumerate(parts):
        if session_re.match(part):
            session_idx = i
            break
    if session_idx is None:
        raise ValueError(
            f"No session component matching {session_re.pattern!r} in path: {path}"
        )

    session = sanitize_id(parts[session_idx], "ses-") 

    return f"{global_id}_{session}"

def sanitize_id(id, regex):
    return id.replace(regex, "")

def sanitize_dataset(id):
    return id.split('_')[0][2:]

# Map a list of scan paths to a sorted, de-duplicated list of participant ids
def portion_to_participant_ids(paths):
    ids = {path_to_participant_id(p) for p in paths}
    return sorted(ids)

def main():
    parser = argparse.ArgumentParser(
        description="Convert split.json into per-split participants TSV lists"
    )
    parser.add_argument("--split-json", required=True, help="Path to split.json.")
    parser.add_argument(
        "--output-dir", default=".", help="Directory for the output <split>.tsv files."
    )
    args = parser.parse_args()

    splits = ["train", "val"]
    split_data = load_json(args.split_json)
    os.makedirs(args.output_dir, exist_ok=True)

    for split in splits:
        file_paths = split_data[0][split]
        ids = portion_to_participant_ids(file_paths)
        out_path = os.path.join(args.output_dir, f"{split}.tsv")
        pd.DataFrame({"participant_id": ids}).to_csv(out_path, sep="\t", index=False)
        print(f"{split}: {len(file_paths)} scans -> {len(ids)} subjects -> {out_path}")


if __name__ == "__main__":
    main()