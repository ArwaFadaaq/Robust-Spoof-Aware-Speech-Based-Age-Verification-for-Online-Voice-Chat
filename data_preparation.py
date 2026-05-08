# -*- coding: utf-8 -*-
"""
Data Preparation Module

This module prepares file-level and segment-level data for the age verification
pipeline. It loads dataset metadata, builds unified file manifests, loads and
combines file/segment manifests, filters and caps segment-level data, enriches
manifests with gender information, saves final train/validation/test splits,
and optionally prepares local runtime copies for faster training.

Overview
--------
The pipeline requires structured data before model training can begin. This
module supports the following steps:

1. Metadata loading
   Dataset-specific metadata from Common Voice, MyST, and VoxCeleb is loaded
   and normalized into a shared schema.

2. File manifest construction
   Speaker split files are joined with dataset metadata to create a unified
   file-level manifest for preprocessing.

3. Manifest loading
   File-level and segment-level manifest CSV files can be loaded and combined
   from one or more directories.

4. Segment analysis and filtering
   Segment manifests are summarized at speaker level. Speakers with too few
   clean segments can be removed, and speakers with many segments can be capped
   to reduce imbalance.

5. Gender enrichment
   Gender labels are extracted from speaker-level split files and merged into
   either file-level or segment-level manifests.

6. Final split saving
   Combined manifests are separated back into train, validation, and test CSV
   files. Segment-level manifests also print summary tables by dataset source,
   age class, and gender.

Inputs
------
- Speaker split CSV files containing speaker_id, split, dataset_source, and
  optionally gender
- Metadata files for Common Voice, MyST, and/or VoxCeleb
- File-level or segment-level manifest CSV files generated during preprocessing
- Final train/validation/test CSV files for local runtime preparation

Assumptions
-----------
- Speaker IDs are consistent between split files and metadata
- MyST speaker IDs are zero-padded when needed
- Splits are speaker-disjoint
- Metadata paths are provided for all dataset sources referenced in the split files
- Segment manifests contain segment_id, speaker_id, dataset_source, split, pool,
  mapped_age_class, seg_path, and is_clipped
- File manifests contain file_id, speaker_id, dataset, split, pool, and audio path
  information

Outputs
-------
- Unified file manifests ready for preprocessing
- Combined file-level or segment-level manifests
- Filtered and capped segment manifests
- Gender-enriched manifests
- Final train/validation/test CSV splits
"""

import os
import glob
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
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
# Manifest Loading
# =========================================================

def load_manifests(manifest_dirs, manifest_type="segment"):
    """
    Load and combine manifest files from multiple directories.

    This function supports loading both segment-level and file-level manifests
    depending on the selected manifest_type.

    Parameters
    ----------
    manifest_dirs : list of str
        List of directories containing manifest CSV files.
    manifest_type : str, default="segment"
        Type of manifest to load:
        - "segment" → loads segment-level manifests (segment_manifest_*.csv)
        - "file"    → loads file-level manifests (file_manifest*.csv)

    Returns
    -------
    pd.DataFrame
        Combined manifest where each row represents one segment or one file.

    Raises
    ------
    ValueError
        If no matching manifest files are found or if an invalid type is provided.
    """

    all_files = []

    # Select file pattern based on manifest type
    if manifest_type == "segment":
        pattern = "segment_manifest_*.csv"
    elif manifest_type == "file":
        pattern = "file_manifest*.csv"
    else:
        raise ValueError("manifest_type must be 'segment' or 'file'")

    # Collect all matching manifest CSV files from the given directories
    for manifest_dir in manifest_dirs:
        csv_files = glob.glob(
            os.path.join(manifest_dir, "**", pattern),
            recursive=True
        )
        all_files.extend(csv_files)

    # Ensure that at least one file was found
    if not all_files:
        raise ValueError(f"No {manifest_type} manifest CSV files were found.")

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


def add_gender_from_splits(manifest_df, role_inventory_paths):
    """
    Enrich a manifest with gender information.

    This function extracts gender labels from speaker-level split files
    and merges them into the input manifest based on speaker_id and dataset source.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Segment-level or file-level manifest.
    role_inventory_paths : list of str
        Paths to speaker split CSV files containing speaker_id, dataset_source,
        and optionally gender.

    Returns
    -------
    pd.DataFrame
        Manifest with an added 'gender' column. Missing values are filled
        with 'unknown'.
    """

    manifest_df = manifest_df.copy()

    # Use dataset_source if available, otherwise use dataset.
    dataset_col = "dataset_source" if "dataset_source" in manifest_df.columns else "dataset"

    split_parts = []

    for path in role_inventory_paths:
        df = pd.read_csv(path, dtype={"speaker_id": str})
        df["speaker_id"] = df["speaker_id"].astype(str).str.strip()

        # Ensure MyST speaker IDs are zero-padded for consistency
        if "dataset_source" in df.columns:
            myst_mask = df["dataset_source"] == "myst"
            df.loc[myst_mask, "speaker_id"] = df.loc[myst_mask, "speaker_id"].str.zfill(6)

        # If gender is missing in the split file, assign 'unknown'
        if "gender" not in df.columns:
            df["gender"] = "unknown"

        split_parts.append(
            df[["speaker_id", "dataset_source", "gender"]].drop_duplicates()
        )

    # Build a unified speaker → gender mapping
    gender_map = pd.concat(split_parts, ignore_index=True).drop_duplicates(
        subset=["speaker_id", "dataset_source"]
    )

    gender_map = gender_map.rename(columns={"dataset_source": dataset_col})

    # Merge gender into the input manifest
    # We use LEFT JOIN (not inner) to avoid dropping any rows
    # in case some speakers do not have gender annotations.
    manifest_df = manifest_df.merge(
        gender_map,
        on=["speaker_id", dataset_col],
        how="left"
    )

    manifest_df["gender"] = manifest_df["gender"].fillna("unknown")

    return manifest_df


def save_final_splits(manifest_df, out_dir, dataset_name="real_clean"):
    """
    Split a combined manifest into train, validation, and test CSV files.

    For segment-level manifests, this function also prints summary tables
    grouped by dataset source, age class, and gender.
    For file-level manifests, it only saves the split CSV files.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Combined segment-level or file-level manifest.
    out_dir : str
        Directory where final split CSV files will be saved.
    dataset_name : str, default="real_clean"
        Name used in the output files.

    Returns
    -------
    dict
        Dictionary containing the saved train, validation, and test splits.
    """

    os.makedirs(out_dir, exist_ok=True)

    final_splits = {}

    split_map = {
        "train": f"train_{dataset_name}",
        "val": f"val_{dataset_name}",
        "test": f"test_{dataset_name}",
    }

    # Segment manifests contain segment_id; file manifests do not.
    is_segment_manifest = "segment_id" in manifest_df.columns

    for split_name, output_name in split_map.items():

        # Select only rows belonging to the current split.
        split_df = manifest_df[
            manifest_df["split"] == split_name
        ].copy()

        if split_df.empty:
            print(f"\nWarning: {output_name} is empty.")
            final_splits[output_name] = split_df
            continue

        # Save final split.
        out_path = os.path.join(out_dir, f"{output_name}.csv")
        split_df.to_csv(out_path, index=False)

        final_splits[output_name] = split_df

        # Print summary only for segment-level manifests.
        if is_segment_manifest:
            summary = (
                split_df
                .groupby(["dataset_source", "mapped_age_class", "gender"], dropna=False)
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
