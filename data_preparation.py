# -*- coding: utf-8 -*-
"""
Data Preparation Module

This module prepares file-level and segment-level data for the age verification pipeline.
It loads speaker splits and dataset-specific metadata, constructs a unified file manifest,
and processes segment-level data to produce the final clean datasets.

Overview
--------
The pipeline requires structured data before model training can begin. This module performs
the following steps:

1. Speaker splits
   CSV files define which speakers belong to train, validation, and test splits.
   Each file corresponds to a specific data pool (e.g., adult_real_candidates).
   Splits are created at the speaker level to prevent data leakage.

2. File-level metadata
   Each dataset source provides metadata linking speakers to audio files.
   Since formats differ across datasets, they are normalized into a shared schema.

3. File manifest construction
   A unified file_manifest.csv is created, where each row represents one audio file
   with all attributes required for preprocessing.

4. Segment-level processing
   Segment manifests are loaded and analyzed to compute the number of segments per speaker.
   Speakers with fewer than a minimum threshold are removed, while speakers with many
   segments are randomly capped to ensure consistency and reduce bias.

5. Final dataset construction
   The filtered segments are used to build the final datasets:
   train_real_clean, val_real_clean, and test_real_clean.
   These splits are checked for balance across age class, dataset source, and speaker distribution.

Inputs
------
- Speaker split CSV files containing speaker_id, split, and dataset_source
- Metadata files for each dataset (Common Voice, MyST, VoxCeleb)
- Segment-level manifest CSV files generated during preprocessing

Assumptions
-----------
- Speaker IDs are consistent between split files and metadata
- Splits are speaker-disjoint (no speaker appears in multiple splits)
- Metadata paths are provided for all datasets referenced in the splits

Output
------
The generated file_manifest contains the following columns:

- dataset           : dataset source (common_voice, myst, voxceleb)
- speaker_id        : unique speaker identifier
- file_id           : unique file identifier
- path              : absolute path to the audio file
- raw_duration_sec  : duration of the audio file in seconds
- split             : train, validation, or test
- pool              : data grouping label (e.g., adult_real_candidates)

Additional outputs include:
- Filtered and capped clean segment manifests
- Final clean dataset splits (train_real_clean, val_real_clean, test_real_clean)
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

np.random.seed(42)

# =========================================================
# Metadata loading
# =========================================================

def load_cv_metadata(cv_meta_path):
    """
    Load and normalize Common Voice metadata.

    Uses clips_path as the file path column and assigns
    the dataset source label.
    """
    df = pd.read_csv(cv_meta_path, low_memory=False)

    df = df.rename(columns={"client_id": "speaker_id"})
    df["path"] = df["clips_path"]
    df["dataset"] = "common_voice"

    return df[["speaker_id", "dataset", "path", "duration_sec"]]


def load_myst_metadata(myst_meta_path):
    """
    Load and normalize MyST metadata.

    Adds dataset source label and returns standardized columns.
    """
    df = pd.read_csv(myst_meta_path, dtype={"speaker_id": str})
    df["speaker_id"] = df["speaker_id"].astype(str).str.strip().str.zfill(6)
    df = df.rename(columns={"full_path": "path"})
    df["dataset"] = "myst"
    return df[["speaker_id", "dataset", "path", "duration_sec"]]


def load_voxceleb_metadata(vox_meta_path):
    """
    Load and normalize VoxCeleb metadata.

    Adds dataset source label and returns standardized columns.
    """
    df = pd.read_csv(vox_meta_path)
    df = df.rename(columns={"full_path": "path"})
    df["dataset"] = "voxceleb"
    return df[["speaker_id", "dataset", "path", "duration_sec"]]


def load_metadata(metadata_paths):
    """
    Load and combine metadata from available dataset sources ONLY.

    Instead of forcing all datasets, we read only what is provided.

    Example:
    metadata_paths = {
        "common_voice": "...",
        "myst": "...",
        "voxceleb": "..."
    }
    """
    loaders = {
        "common_voice": load_cv_metadata,
        "myst":         load_myst_metadata,
        "voxceleb":     load_voxceleb_metadata,
    }

    all_meta = []

    # Loop over provided datasets only
    for dataset_name, meta_path in metadata_paths.items():
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        meta_df = loaders[dataset_name](meta_path)
        all_meta.append(meta_df)

    if not all_meta:
        raise ValueError("No metadata paths were provided.")

    return pd.concat(all_meta, ignore_index=True)


# =========================================================
# File manifest builder
# =========================================================

def build_file_manifest(role_inventory_paths, metadata_paths, out_path):
    """
    Build the unified file manifest.

    Processing steps
    ----------------
    1. Load file-level metadata from ONLY provided dataset sources.
    2. Iterate over all role inventory files.
    3. Validate that all required datasets have metadata paths.
    4. Filter metadata to keep only files whose speakers are listed in the current inventory.
    5. Merge speaker information with file-level metadata.
    6. Construct the pool label based on the role name.
    7. Concatenate all pool-specific manifests.
    8. Generate file_id and rename columns to match manifest format.
    9. Save the final file_manifest.csv.

    Returns
    -------
    pd.DataFrame
        Final file manifest ready for preprocessing.
    """

    # Step 1: Load metadata once (ONLY what user provided)
    all_meta = load_metadata(metadata_paths)

    manifest_parts = []

    # Step 2: Process each inventory file
    for role_inventory_path in role_inventory_paths:
        speakers = pd.read_csv(role_inventory_path, dtype={"speaker_id": str})
        speakers["speaker_id"] = speakers["speaker_id"].astype(str).str.strip()

        if "dataset_source" in speakers.columns:
            myst_mask = speakers["dataset_source"] == "myst"
            speakers.loc[myst_mask, "speaker_id"] = speakers.loc[myst_mask, "speaker_id"].str.zfill(6)

        # Step 3: Validate all required datasets have metadata paths
        available_datasets = set(metadata_paths.keys())
        required_datasets  = set(speakers["dataset_source"].dropna().unique())
        missing_datasets   = required_datasets - available_datasets
        if missing_datasets:
            raise ValueError(
                f"Missing metadata paths for datasets found in inventory: {sorted(missing_datasets)}"
            )

        target_ids = set(speakers["speaker_id"])

        # Step 4: Filter by target speakers
        filtered_meta = all_meta[all_meta["speaker_id"].isin(target_ids)].copy()

        # Step 5: Merge speaker info
        file_list = filtered_meta.merge(
            speakers[["speaker_id", "split", "dataset_source"]].rename(columns={"dataset_source": "dataset"}),
            on=["speaker_id", "dataset"],
            how="inner",
        )

        assert len(file_list) == len(filtered_meta), f"Merge mismatch in {role_inventory_path}"

        # Step 6: Build pool
        role_name = os.path.basename(role_inventory_path).replace("_split.csv", "")
        file_list["pool"] = role_name

        manifest_parts.append(file_list)

    # Step 7: Combine all pools
    file_manifest = pd.concat(manifest_parts, ignore_index=True)

    # Step 8: Rename columns and generate file_id
    file_manifest = file_manifest.rename(columns={
        "duration_sec": "raw_duration_sec"
    }).copy()

    file_manifest["file_id"] = [
        f"{ds}_{spk}_{i:06d}"
        for i, (ds, spk) in enumerate(
            zip(file_manifest["dataset"], file_manifest["speaker_id"]),
            start=1
        )
    ]

    # Select final columns
    file_manifest = file_manifest[
        ["dataset", "speaker_id", "file_id", "path", "raw_duration_sec", "split", "pool"]
    ]

    # Step 9: Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    file_manifest.to_csv(out_path, index=False)

    # Summary
    print(f"Total speakers: {file_manifest['speaker_id'].nunique()}")
    print(f"Total files:    {len(file_manifest)}")
    print(f"Saved: {out_path}")

    return file_manifest


# =========================================================
# Segment Analysis and Filtering
# =========================================================

def load_segment_manifests(manifest_dirs):
    """
    Load and combine segment-level manifest files from multiple directories.

    Parameters
    ----------
    manifest_dirs : list of str
        List of directories containing segment manifest CSV files.

    Returns
    -------
    pd.DataFrame
        Combined segment-level manifest where each row represents one segment.
    """

    all_files = []

    # Collect all segment manifest CSV files from the given directories
    for manifest_dir in manifest_dirs:
        csv_files = glob.glob(
            os.path.join(manifest_dir, "**", "segment_manifest_*.csv"),
            recursive=True
        )
        all_files.extend(csv_files)

    if not all_files:
        raise ValueError("No segment manifest CSV files were found.")

    # Load each manifest and combine them into one DataFrame
    dfs = [pd.read_csv(path) for path in all_files if os.path.isfile(path)]

    return pd.concat(dfs, ignore_index=True)


def summarize_segment_distribution(segment_manifest, plot=True, cap_line=14):
    """
    Summarize segment counts per speaker.

    Computes speaker-level segment statistics and optionally plots a horizontal
    boxplot to visualize the distribution.

    Parameters
    ----------
    segment_manifest : pd.DataFrame
        Segment-level manifest where each row represents one audio segment.
    plot : bool, default=True
        Whether to display a boxplot.
    cap_line : int or None, default=14
        Optional reference line showing the selected segment cap.

    Returns
    -------
    pd.DataFrame
        Speaker-level statistics with one row per speaker.
    """

    # Validate required columns
    required_cols = ["speaker_id", "segment_id", "dataset_source", "mapped_age_class"]
    for col in required_cols:
        if col not in segment_manifest.columns:
            raise ValueError(f"Missing column: {col}")

    # Count the number of segments available for each speaker
    speaker_stats = (
        segment_manifest
        .groupby(["speaker_id", "dataset_source", "mapped_age_class"], dropna=False)
        .agg(num_segments=("segment_id", "count"))
        .reset_index()
    )

    # Print descriptive statistics to inspect imbalance and outliers
    summary = speaker_stats["num_segments"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
    )

    print("Segments per speaker summary:")
    print(summary)

    if plot:
        # Plot the main distribution without extreme outliers
        plt.figure(figsize=(6, 4))

        plt.boxplot(
            speaker_stats["num_segments"],
            showfliers=False,
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="#DCEEFF", linewidth=1),
            medianprops=dict(color="red", linewidth=2)
        )

        vals = speaker_stats["num_segments"]

        Q1 = vals.quantile(0.25)
        Q3 = vals.quantile(0.75)
        IQR = Q3 - Q1

        upper_whisker = min(Q3 + 1.5 * IQR, vals.max())

        step = 5 if upper_whisker > 50 else 2
        plt.xticks(np.arange(0, int(upper_whisker)+step, step))

        plt.yticks([])

        # Add a light grid to make segment counts easier to read
        plt.grid(axis="x", linestyle="--", alpha=0.4, linewidth=0.5)

        # Add an optional reference line for the selected cap value
        if cap_line is not None:
            plt.axvline(x=cap_line, linestyle="--")

        plt.title("Segments per Speaker (Boxplot without outliers)")
        plt.xlabel("Number of Segments")

        plt.tight_layout()
        plt.show()

    return speaker_stats


def filter_and_cap_segments(segment_manifest, min_segments=8, max_segments=14, seed=42):
    """
    Remove speakers with too few segments and cap segments per speaker.

    Speakers with segment counts less than min_segments are removed.
    Speakers with more than max_segments are randomly sampled down to max_segments.

    Parameters
    ----------
    segment_manifest : pd.DataFrame
        Segment-level manifest where each row represents one audio segment.
    min_segments : int, default=8
        Remove speakers with fewer than min_segments.
    max_segments : int, default=14
        Maximum number of segments to keep per speaker.
    seed : int, default=42
        Random seed for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Filtered and capped segment manifest.
    """
    
    original_segments = len(segment_manifest)
    original_speakers = segment_manifest["speaker_id"].nunique()

    # Remove clipped segments completely (treat them as noisy)
    if "is_clipped" in segment_manifest.columns:
        segment_manifest = segment_manifest[
            segment_manifest["is_clipped"] == False
        ].copy()

    clean_segments = len(segment_manifest)
    clean_speakers = segment_manifest["speaker_id"].nunique()

    # Count segments per speaker after removing clipped
    speaker_counts = (
        segment_manifest
        .groupby("speaker_id")
        .size()
        .reset_index(name="num_segments")
    )

    valid_speakers = speaker_counts[
        speaker_counts["num_segments"] >= min_segments
    ]["speaker_id"]

    filtered_manifest = segment_manifest[
        segment_manifest["speaker_id"].isin(valid_speakers)
    ].copy()

    capped_manifest = (
        filtered_manifest
        .groupby("speaker_id", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=min(len(x), max_segments),
                random_state=seed
            )
        )
        .reset_index(drop=True)
    )

    print("Original segment manifest:")
    print("Segments:", original_segments)
    print("Speakers:", original_speakers)

    print("\nAfter removing clipped segments:")
    print("Segments:", clean_segments)
    print("Speakers:", clean_speakers)

    print(f"\nAfter removing speakers with < {min_segments} clean segments:")
    print("Segments:", len(filtered_manifest))
    print("Speakers:", filtered_manifest["speaker_id"].nunique())

    print(f"\nAfter capping to max {max_segments} clean segments per speaker:")
    print("Segments:", len(capped_manifest))
    print("Speakers:", capped_manifest["speaker_id"].nunique())

    print("\nClean segments per speaker after filtering and cap:")
    print(capped_manifest.groupby("speaker_id").size().describe())

    return capped_manifest


def build_final_real_clean_splits(capped_manifest, out_dir):
    """
    Create final real-clean train, validation, and test datasets.

    This function splits the filtered and capped segment manifest into
    train_real_clean, val_real_clean, and test_real_clean based on the
    existing split column. It saves each split as a CSV file and prints
    summary tables grouped by dataset source and age class.

    Parameters
    ----------
    capped_manifest : pd.DataFrame
        Filtered and capped segment-level manifest.
    out_dir : str
        Directory where final split CSV files will be saved.

    Returns
    -------
    dict
        Dictionary containing train_real_clean, val_real_clean, and test_real_clean.
    """

    # Validate required columns
    required_cols = ["speaker_id", "segment_id", "split", "mapped_age_class", "dataset_source"]
    for col in required_cols:
        if col not in capped_manifest.columns:
            raise ValueError(f"Missing column: {col}")

    os.makedirs(out_dir, exist_ok=True)

    final_splits = {}

    split_map = {
        "train": "train_real_clean",
        "val": "val_real_clean",
        "test": "test_real_clean"
    }

    for split_name, output_name in split_map.items():

        # Select only rows belonging to the current split
        split_df = capped_manifest[
            capped_manifest["split"] == split_name
        ].copy()

        if split_df.empty:
            print(f"\nWarning: {output_name} is empty.")
            final_splits[output_name] = split_df
            continue

        # Save final split
        out_path = os.path.join(out_dir, f"{output_name}.csv")
        split_df.to_csv(out_path, index=False)

        final_splits[output_name] = split_df

        summary = (
            split_df
            .groupby(["dataset_source", "mapped_age_class"])
            .agg(
                total_segments=("segment_id", "count"),
                total_speakers=("speaker_id", "nunique")
            )
            .reset_index()
        )

        print("\n" + "=" * 60)
        print(output_name.upper())
        print("=" * 60)
        display(summary)
        print(f"Saved to: {out_path}")

    return final_splits
