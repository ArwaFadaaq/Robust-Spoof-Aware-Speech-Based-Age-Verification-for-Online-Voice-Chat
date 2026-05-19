"""
Local Runtime Data Utilities

This module provides utilities for preparing large audio datasets for
efficient training and evaluation in Google Colab or similar environments.

Overview
--------
The workflow is designed to reduce Google Drive I/O bottlenecks by:
1. Creating compressed tar archives for reuse
2. Extracting archives into local runtime storage
3. Updating CSV manifests so audio paths point to local files

Main Features
-------------
- Create reusable tar archives
- Extract archives locally
- Automatically update manifest paths
- Prepare train/validation/test manifests for local runtime

Typical Usage
-------------

1. Create reusable tar archive:
    create_tar(...)

2. Load archive in future sessions:
    load_archive_to_local(...)

Notes
-----
- Designed for speech/audio experiments
- Optimized for Google Colab workflows
- Assumes manifests contain absolute Drive paths
"""

import os
import shutil
import subprocess
import pandas as pd


# =========================================================
# Update Manifest Paths
# =========================================================

def update_manifest_paths(
    csv_path,
    out_csv_path,
    base_map,
    audio_col="seg_path"
):
    """
    Replace Drive audio paths with local runtime paths.

    Parameters
    ----------
    csv_path : str
        Original CSV manifest path.

    out_csv_path : str
        Output path for updated local manifest.

    base_map : dict
        Dictionary mapping Drive base directories
        to local runtime directories.

    audio_col : str
        Column containing audio paths.

    Returns
    -------
    str
        Path to updated local CSV manifest.
    """

    # Load manifest
    df = pd.read_csv(csv_path)

    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df)}")

    # Initialize local paths
    df["local_path"] = df[audio_col]

    # Replace Drive paths → local runtime paths
    for drive_base_dir, local_base_dir in base_map.items():

        mask = df["local_path"].str.startswith(
            drive_base_dir,
            na=False
        )

        df.loc[mask, "local_path"] = (df.loc[mask, "local_path"]
            .str.replace(drive_base_dir, local_base_dir, regex=False))

    print("Updated paths to local runtime.")

    # Save updated manifest
    df.to_csv(out_csv_path, index=False)

    print(f"Saved local manifest → {out_csv_path}")

    return out_csv_path


# =========================================================
# Create Tar Archive
# =========================================================

def create_tar(source_dirs, tar_path):
    """
    Create a tar archive from multiple source directories.

    Each directory is added independently to the same archive,
    allowing folders from different parent locations to be stored
    together without requiring a shared root directory.

    Parameters
    ----------
    source_dirs : list
        List of folder paths to include in the archive.

    tar_path : str
        Output tar archive path.

    Returns
    -------
    str
        Path to the created tar archive.
    """

    # Remove existing archive if present
    if os.path.exists(tar_path):
        os.remove(tar_path)

    # -----------------------------------------------------
    # Add folders to archive
    # -----------------------------------------------------

    for i, source_dir in enumerate(source_dirs):

        # Extract folder name
        folder_name = os.path.basename(source_dir)

        # Parent directory
        parent_dir = os.path.dirname(source_dir)

        # Base tar command
        cmd = [
            "tar",
            "--warning=no-file-changed"
        ]

        # First folder → create archive
        if i == 0:
            cmd += ["-cf"]

        # Remaining folders → append
        else:
            cmd += ["-rf"]

        # Add archive arguments
        cmd += [
            tar_path,
            "-C",
            parent_dir,
            folder_name
        ]

        # Run tar command
        subprocess.run(cmd, check=True)

        print(f"Added → {source_dir}")

    print(f"\nCreated archive → {tar_path}")

    return tar_path


# =========================================================
# Copy File
# =========================================================

def copy_file(src, dst):
    """
    Copy file from source to destination.

    Parameters
    ----------
    src : str
        Source file path.

    dst : str
        Destination file path.

    Returns
    -------
    str
        Destination path.
    """

    # Create destination directory if missing
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Copy file
    shutil.copy2(src, dst)

    print(f"Copied file:")
    print(f"  From → {src}")
    print(f"  To   → {dst}")

    return dst


# =========================================================
# Extract Tar Archive
# =========================================================

def extract_tar(tar_path, extract_dir="/content/audio_data",
                members=None):
    """
    Extract tar archive locally.

    Parameters
    ----------
    tar_path : str
        Tar archive path.

    extract_dir : str
        Directory where archive will be extracted.

    members : list or None
        Specific folders/files to extract from the archive.
        If None, the full archive is extracted.
    """

    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)

    cmd = [
        "tar",
        "-xf",
        tar_path,
        "-C",
        extract_dir
    ]

    if members is not None:
        cmd += members

    # Extract archive
    subprocess.run(cmd, check=True)

    print(f"Extracted archive → {extract_dir}")


# =========================================================
# Load Archive + Update Manifests
# =========================================================

def load_archive_to_local(
    drive_tar_path,
    csv_inputs,
    base_map,
    local_tar_path="/content/data_archive.tar",
    extract_dir="/content/audio_data",
    local_manifest_dir="/content/local_manifests",
    members=None
):
    """
    Load compressed archive locally and prepare manifests.

    Workflow
    --------
    1. Copy archive from Drive
    2. Extract archive locally
    3. Update manifest paths
    4. Return local manifest paths

    Parameters
    ----------
    drive_tar_path : str
        Archive path stored in Drive.

    csv_inputs : dict
        Dictionary where each key is an audio path column name and each value is a list of CSV manifest paths.

    base_map : dict
        Dictionary mapping Drive base directories
        to local runtime directories.

    local_tar_path : str
        Temporary local tar path.

    extract_dir : str
        Local extraction directory.

    local_manifest_dir : str
        Directory where updated manifests will be saved.

    members : list or None
        Specific folders/files to extract from the archive.
        If None, the full archive is extracted.

    Returns
    -------
    list
        Updated local CSV manifest paths.
    """

    # -----------------------------------------------------
    # Copy archive locally
    # -----------------------------------------------------

    copy_file(
        src=drive_tar_path,
        dst=local_tar_path
    )

    # -----------------------------------------------------
    # Extract archive locally
    # -----------------------------------------------------

    extract_tar(
        tar_path=local_tar_path,
        extract_dir=extract_dir,
        members=members
    )

    # -----------------------------------------------------
    # Update manifests
    # -----------------------------------------------------

    os.makedirs(local_manifest_dir, exist_ok=True)

    local_csv_paths = []

    for audio_col, csv_list in csv_inputs.items():

        for csv_path in csv_list:

            # Create local manifest name
            filename = os.path.basename(csv_path)

            out_csv_path = os.path.join(
                local_manifest_dir,
                filename.replace(".csv", "_local.csv")
            )

            # Update paths
            local_csv = update_manifest_paths(
                csv_path=csv_path,
                out_csv_path=out_csv_path,
                base_map=base_map,
                audio_col=audio_col
            )

            local_csv_paths.append(local_csv)

    print("\nArchive loaded successfully.")
    print("Local manifests ready.")

    return local_csv_paths
