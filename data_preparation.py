# -*- coding: utf-8 -*-
"""
Dataset Preparation Module

This module prepares processed audio datasets for the age verification
experiments.

Supported datasets
------------------
1. Common Voice
   - Metadata provides age and gender labels.
   - Age labels are converted into a binary setting: child vs adult.

2. MyST
   - All samples are treated as child speech.
   - Speaker ID is extracted from the folder structure.

Processing overview
-------------------
1. Load dataset metadata or scan audio files.
2. Preserve the original train/valid/test split.
3. Build full paths to audio recordings.
4. Apply the audio preprocessing pipeline.
5. For short recordings:
   - Segment each recording into fixed-length windows.
   - Save the generated segments to disk.
   - Store metadata for each saved segment.
6. For long recordings (duration >= threshold):
   - Apply preprocessing only.
   - Save the processed full audio without segmentation.
   - Store metadata in a separate CSV file.
7. Support resuming from previous runs:
   - Previously processed files are detected from saved metadata CSVs.
   - Progress is periodically saved to disk.
"""

import os
import sys
import kagglehub
import pandas as pd
import torchaudio
from tqdm import tqdm

from preprocessing import (
    preprocess_by_dataset,
    preprocess_full_audio,
    save_segment_as_wav,
)


# Kaggle identifier for Common Voice
COMMON_VOICE_NAME = "mozillaorg/common-voice"


# =========================================================
# Utility Functions
# =========================================================
def show_metadata_distribution(df: pd.DataFrame, title: str) -> None:
    """
    Display basic statistics for a metadata dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input metadata dataframe.
    title : str
        Title describing the current stage of processing.
    """
    print(f"\n=== {title} ===")

    if "age" in df.columns:
        print("\nAge distribution:")
        print(df["age"].value_counts(dropna=False))

    if "gender" in df.columns:
        print("\nGender distribution:")
        print(df["gender"].value_counts(dropna=False))

    if "split" in df.columns:
        print("\nSplit distribution:")
        print(df["split"].value_counts(dropna=False))

    print("\nTotal rows:", len(df))


# =========================================================
# Common Voice: Load metadata
# =========================================================
def load_common_voice_metadata() -> tuple[pd.DataFrame, str]:
    """
    Download Common Voice and load metadata for all official splits.

    Returns
    -------
    df : pd.DataFrame
        Combined metadata dataframe for train, valid, and test splits.
    dataset_path : str
        Root path of the downloaded Common Voice dataset.
    """
    dataset_path = kagglehub.dataset_download(COMMON_VOICE_NAME)
    print("Common Voice dataset path:", dataset_path)

    split_files = {
        "train": "cv-valid-train.csv",
        "valid": "cv-valid-dev.csv",
        "test": "cv-valid-test.csv",
    }

    frames = []

    for split_name, file_name in split_files.items():
        csv_path = os.path.join(dataset_path, file_name)

        if not os.path.exists(csv_path):
            print(f"Warning: metadata file not found: {csv_path}")
            continue

        df_split = pd.read_csv(csv_path)
        df_split["split"] = split_name
        frames.append(df_split)

    if not frames:
        raise ValueError("No Common Voice metadata files were found.")

    df = pd.concat(frames, ignore_index=True)
    return df, dataset_path


# =========================================================
# Common Voice: Clean metadata
# =========================================================
def clean_common_voice_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Common Voice metadata and convert age labels to binary classes.

    Processing steps
    ----------------
    1. Remove rows with missing age or gender.
    2. Remove rows where gender is labeled as 'other'.
    3. Convert age labels into:
       - child  -> teens
       - adult  -> all remaining age categories

    Parameters
    ----------
    df : pd.DataFrame
        Raw Common Voice metadata dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned metadata dataframe.
    """
    df = df.dropna(subset=["age", "gender"]).copy()
    df = df[df["gender"] != "other"]

    df["age"] = df["age"].apply(
        lambda x: "child" if x == "teens" else "adult"
    )

    return df


# =========================================================
# Common Voice: Build audio paths
# =========================================================
def add_common_voice_audio_paths(
    df: pd.DataFrame,
    dataset_path: str
) -> pd.DataFrame:
    """
    Build full audio paths for Common Voice recordings based on split.

    Parameters
    ----------
    df : pd.DataFrame
        Common Voice metadata dataframe.
    dataset_path : str
        Root path of the Common Voice dataset.

    Returns
    -------
    pd.DataFrame
        Metadata dataframe with:
        - audio_path : full path to the audio file
        - file_name  : original file name
        - speaker_id : placeholder value if unavailable
    """
    split_to_folder = {
        "train": "cv-valid-train",
        "valid": "cv-valid-dev",
        "test": "cv-valid-test",
    }

    df = df.copy()

    df["audio_path"] = df.apply(
        lambda row: os.path.join(
            dataset_path,
            split_to_folder[row["split"]],
            row["filename"],
        ),
        axis=1,
    )

    df["file_name"] = df["filename"].apply(os.path.basename)

    if "speaker_id" not in df.columns:
        df["speaker_id"] = "unknown"

    return df


# =========================================================
# MyST: Scan audio files
# =========================================================
def load_myst_audio_paths(
    myst_root_dir: str,
    splits: list[str] | None = None
) -> pd.DataFrame:
    """
    Scan the MyST dataset directory and collect audio file paths.

    Assumptions
    -----------
    1. The dataset is organized into train / development / test folders.
    2. All samples are treated as child speech.
    3. Gender information is not used.
    4. Audio recordings are stored as .flac files.
    5. Speaker ID is extracted from the student folder.

    Expected structure
    ------------------
    <root>/<partition>/<student_id>/<session_id>/<file>.flac

    Parameters
    ----------
    myst_root_dir : str
        Root directory of the MyST dataset.
    splits : list[str] | None, optional
        Dataset splits to load. Supported values are
        ["train", "valid", "test"].

        If None, all splits are loaded.

    Returns
    -------
    pd.DataFrame
        Dataframe containing:
        - audio_path  : path to the audio file
        - file_name   : original file name
        - speaker_id  : extracted student ID
        - age         : fixed label "child"
        - gender      : placeholder value
        - split       : dataset split (train/valid/test)
    """
    records = []

    split_map = {
        "train": "train",
        "valid": "development",
        "test": "test",
    }

    if splits is None:
        splits = ["train", "valid", "test"]

    for split_name in splits:
        folder_name = split_map[split_name]
        split_dir = os.path.join(myst_root_dir, folder_name)

        if not os.path.exists(split_dir):
            print(f"Warning: split folder not found: {split_dir}")
            continue

        for root, _, files in os.walk(split_dir):
            for file_name in files:
                if not file_name.lower().endswith(".flac"):
                    continue

                audio_path = os.path.join(root, file_name)

                rel_dir = os.path.relpath(root, split_dir)
                parts = rel_dir.split(os.sep)

                student_id = parts[0] if len(parts) > 0 else "unknown"

                records.append({
                    "audio_path": audio_path,
                    "file_name": file_name,
                    "speaker_id": student_id,
                    "age": "child",
                    "gender": "NA",
                    "split": split_name,
                })

    df = pd.DataFrame(records)

    print("Total MyST audio files:", len(df))

    if len(df) > 0:
        print("\nMyST split distribution:")
        print(df["split"].value_counts())

    return df


# =========================================================
# Shared Processing: Segment generation, saving, and resume
# =========================================================
def process_segments(
    df: pd.DataFrame,
    processed_dir: str,
    dataset_name: str,
    spoof_label: int,
    remove_internal_silence: bool,
    top_db: int,
    seg_sec: float,
    hop_sec: float,
    metadata_csv_path: str | None = None,
    long_files_csv_path: str | None = None,
    long_audio_dir: str | None = None,
    no_segment_min_sec: float = 10.0,
    return_df: bool = True,
    resume: bool = True,
    save_every: int = 100,
) -> pd.DataFrame | None:
    """
    Apply preprocessing, save short files as segments, and save long files
    as full processed audio without segmentation.

    Processing rules
    ----------------
    1. Very short recordings (duration < seg_sec):
       - Skip the file.
       - Do not save any output.

    2. Short recordings (seg_sec <= duration < no_segment_min_sec):
       - Apply dataset-specific preprocessing.
       - Generate fixed-length segments.
       - Save each segment inside its split folder.
       - Store segment-level metadata.

    3. Long recordings (duration >= no_segment_min_sec):
       - Apply the same preprocessing pipeline.
       - Do not perform segmentation.
       - Save the processed full waveform in a separate external folder.
       - Store file-level metadata in a separate CSV.

    Resume behavior
    ---------------
    If resume=True and previous metadata CSV files already exist, the function:
    - Loads previously saved metadata.
    - Collects processed original audio paths from both segment-level and
      long-file metadata.
    - Skips any file whose original audio_path has already been processed.
    - Continues from the remaining unprocessed files only.

    Saving behavior
    ---------------
    To reduce the risk of losing progress during runtime interruption, the
    function periodically saves both metadata CSV files every `save_every`
    newly processed original audio files.

    Duration check behavior
    -----------------------
    For each input audio file, the function first tries to read file metadata
    using `torchaudio.info(...)` in order to estimate the duration without
    loading the full waveform into memory.

    If this step fails for any reason, the function falls back to
    `torchaudio.load(...)` and computes the duration from the loaded waveform.

    Parameters
    ----------
    df : pd.DataFrame
        Input metadata dataframe. Must contain at least:
        - "audio_path"
        - "split"

        Optional columns such as "speaker_id", "file_name", "age", and
        "gender" are also used when available.
    processed_dir : str
        Directory where processed segments will be saved.
    dataset_name : str
        Dataset identifier (e.g. 'cv' or 'myst').
    spoof_label : int
        Spoof label assigned to all saved outputs.
    remove_internal_silence : bool
        Whether to remove internal non-speech regions.
    top_db : int
        Silence threshold used for edge trimming.
    seg_sec : float
        Segment length in seconds.
    hop_sec : float
        Hop length in seconds.
    metadata_csv_path : str | None, optional
        Output CSV path for segment-level metadata.

        If None, a default file is created inside processed_dir.
    long_files_csv_path : str | None, optional
        Output CSV path for long processed files metadata.

        If None, a default file is created inside long_audio_dir.
    long_audio_dir : str | None, optional
        External directory where long processed files will be saved.

        If None, a default folder is created next to processed_dir.
    no_segment_min_sec : float, default=10.0
        Minimum duration threshold above which segmentation is skipped.
    return_df : bool, default=True
        Whether to return the segment-level dataframe.
    resume : bool, default=True
        Whether to continue from previously saved metadata and skip already
        processed audio files.
    save_every : int, default=100
        Number of newly processed original audio files after which progress
        is written to disk.

    Returns
    -------
    pd.DataFrame | None
        Segment-level metadata dataframe if return_df=True, else None.
    """
    os.makedirs(processed_dir, exist_ok=True)

    if long_audio_dir is None:
        parent_dir = os.path.dirname(processed_dir.rstrip("/"))
        long_audio_dir = os.path.join(parent_dir, f"{dataset_name}_long_full_audio")

    os.makedirs(long_audio_dir, exist_ok=True)

    if metadata_csv_path is None:
        metadata_csv_path = os.path.join(
            processed_dir,
            f"{dataset_name}_segments_metadata.csv"
        )

    if long_files_csv_path is None:
        long_files_csv_path = os.path.join(
            long_audio_dir,
            f"{dataset_name}_long_files_processed_no_segmentation.csv"
        )

    os.makedirs(os.path.dirname(metadata_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(long_files_csv_path), exist_ok=True)

    if resume and os.path.exists(metadata_csv_path):
        segments_df = pd.read_csv(metadata_csv_path)
        segment_records = segments_df.to_dict("records")
    else:
        segments_df = pd.DataFrame()
        segment_records = []

    if resume and os.path.exists(long_files_csv_path):
        long_files_df = pd.read_csv(long_files_csv_path)
        long_files_records = long_files_df.to_dict("records")
    else:
        long_files_df = pd.DataFrame()
        long_files_records = []

    processed_audio_paths = set()

    if not segments_df.empty and "audio_path" in segments_df.columns:
        processed_audio_paths.update(
            segments_df["audio_path"].dropna().tolist()
        )

    if not long_files_df.empty and "audio_path" in long_files_df.columns:
        processed_audio_paths.update(
            long_files_df["audio_path"].dropna().tolist()
        )

    print(f"Already processed files: {len(processed_audio_paths)}")

    if resume and len(processed_audio_paths) > 0:
        df = df[~df["audio_path"].isin(processed_audio_paths)].copy()

    df = df.reset_index().rename(columns={"index": "original_idx"})

    print(f"Remaining files to process: {len(df)}")

    segments_per_split = {}
    if not segments_df.empty and "split" in segments_df.columns:
        segments_per_split = segments_df["split"].value_counts().to_dict()

    long_files_count = len(long_files_records)
    newly_processed_count = 0

    pbar = tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}")

    for _, row in pbar:
        audio_path = row["audio_path"]

        try:
            # -----------------------------------------
            # Read duration safely
            # -----------------------------------------
            try:
                if hasattr(torchaudio, "info"):
                    info = torchaudio.info(audio_path)
                    duration_sec = info.num_frames / info.sample_rate
                else:
                    waveform, sample_rate = torchaudio.load(audio_path)
                    duration_sec = waveform.shape[1] / sample_rate
            except Exception:
                waveform, sample_rate = torchaudio.load(audio_path)
                duration_sec = waveform.shape[1] / sample_rate

            if duration_sec < seg_sec:
                continue

            speaker = row.get("speaker_id", "unknown")
            original_file_name = row.get("file_name", os.path.basename(audio_path))
            split_name = row["split"]
            base_file_id = f"{row['original_idx']:06d}"

            # -------------------------------------------------
            # Long files: preprocess only, no segmentation
            # -------------------------------------------------
            if duration_sec >= no_segment_min_sec:
                processed_audio = preprocess_full_audio(
                    audio_path=audio_path,
                    dataset_name=dataset_name,
                )

                if processed_audio.numel() == 0:
                    continue

                full_file_name = (
                    f"{dataset_name}_{split_name}_sp{speaker}_"
                    f"f{base_file_id}_full.wav"
                )
                full_path = os.path.join(long_audio_dir, full_file_name)

                if not os.path.exists(full_path):
                    save_segment_as_wav(
                        segment=processed_audio,
                        output_path=full_path
                    )

                long_files_records.append({
                    "audio_path": audio_path,
                    "processed_path": full_path,
                    "file_name": original_file_name,
                    "speaker_id": speaker,
                    "split": split_name,
                    "dataset": dataset_name,
                    "sample_id": row["original_idx"],
                    "age": row.get("age", None),
                    "gender": row.get("gender", None),
                    "spoof_label": spoof_label,
                    "duration_sec": round(duration_sec, 3),
                    "processing_type": "preprocessed_no_segmentation",
                })

                processed_audio_paths.add(audio_path)
                long_files_count += 1
                newly_processed_count += 1

            # -------------------------------------------------
            # Short files: preprocess + segment
            # -------------------------------------------------
            else:
                segments = preprocess_by_dataset(
                    audio_path=audio_path,
                    dataset_name=dataset_name,
                    remove_internal_silence=remove_internal_silence,
                    top_db=top_db,
                    seg_sec=seg_sec,
                    hop_sec=hop_sec,
                )

                if len(segments) == 0:
                    continue

                split_dir = os.path.join(processed_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)

                for seg_idx, segment in enumerate(segments):
                    segment_file_name = (
                        f"{dataset_name}_{split_name}_sp{speaker}_"
                        f"f{base_file_id}_seg{seg_idx:02d}.wav"
                    )
                    segment_path = os.path.join(split_dir, segment_file_name)

                    if not os.path.exists(segment_path):
                        save_segment_as_wav(
                            segment=segment,
                            output_path=segment_path
                        )

                    segments_per_split[split_name] = (
                        segments_per_split.get(split_name, 0) + 1
                    )

                    segment_records.append({
                        "audio_path": audio_path,
                        "segment_path": segment_path,
                        "file_name": original_file_name,
                        "speaker_id": speaker,
                        "split": split_name,
                        "dataset": dataset_name,
                        "segment_id": seg_idx,
                        "sample_id": row["original_idx"],
                        "age": row.get("age", None),
                        "gender": row.get("gender", None),
                        "spoof_label": spoof_label,
                        "duration_sec": round(duration_sec, 3),
                    })

                processed_audio_paths.add(audio_path)
                newly_processed_count += 1

            if newly_processed_count > 0 and newly_processed_count % save_every == 0:
                pd.DataFrame(segment_records).to_csv(
                    metadata_csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )
                pd.DataFrame(long_files_records).to_csv(
                    long_files_csv_path,
                    index=False,
                    encoding="utf-8-sig"
                )
                pbar.set_postfix(saved=newly_processed_count)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    segments_df = pd.DataFrame(segment_records)
    long_files_df = pd.DataFrame(long_files_records)

    segments_df.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")
    long_files_df.to_csv(long_files_csv_path, index=False, encoding="utf-8-sig")

    print(f"\nTotal saved {dataset_name} segments: {len(segments_df)}")
    print(f"Total saved long processed files: {len(long_files_df)}")
    print("\nSegments per split:")

    for split_name, count in segments_per_split.items():
        print(f"{split_name}: {count}")

    print("\nTotal long processed files:", long_files_count)
    print(f"Segments metadata saved to: {metadata_csv_path}")
    print(f"Long files metadata saved to: {long_files_csv_path}")

    if return_df:
        return segments_df
    return None


# =========================================================
# Public API: Prepare Common Voice
# =========================================================
def prepare_common_voice_dataset(
    processed_dir: str,
    remove_internal_silence: bool = False,
    top_db: int = 30,
    seg_sec: float = 3.0,
    hop_sec: float = 1.5,
    metadata_csv_path: str | None = None,
    long_files_csv_path: str | None = None,
    long_audio_dir: str | None = None,
    no_segment_min_sec: float = 10.0,
    return_df: bool = True,
    resume: bool = True,
    save_every: int = 100,
) -> pd.DataFrame | None:
    """
    Prepare processed Common Voice data.

    Parameters
    ----------
    processed_dir : str
        Directory where processed segments will be saved.
    remove_internal_silence : bool, default=False
        Whether to remove internal non-speech regions.
    top_db : int, default=30
        Silence threshold used for edge trimming.
    seg_sec : float, default=3.0
        Segment length in seconds.
    hop_sec : float, default=1.5
        Hop length in seconds.
    metadata_csv_path : str | None, optional
        Output CSV path for segment-level metadata.
    long_files_csv_path : str | None, optional
        Output CSV path for long processed files metadata.
    long_audio_dir : str | None, optional
        External folder for long processed full audio files.
    no_segment_min_sec : float, default=10.0
        Duration threshold for skipping segmentation.
    return_df : bool, default=True
        Whether to return the segment-level dataframe.
    resume : bool, default=True
        Whether to continue from previously saved metadata.
    save_every : int, default=100
        Number of newly processed original audio files after which progress
        is written to disk.

    Returns
    -------
    pd.DataFrame | None
        Segment-level metadata dataframe for Common Voice.
    """
    df, dataset_path = load_common_voice_metadata()
    show_metadata_distribution(df, "COMMON VOICE - BEFORE CLEANING")

    df = clean_common_voice_metadata(df)
    show_metadata_distribution(df, "COMMON VOICE - AFTER CLEANING")

    df = add_common_voice_audio_paths(df, dataset_path)

    return process_segments(
        df=df,
        processed_dir=processed_dir,
        dataset_name="cv",
        spoof_label=0,
        remove_internal_silence=remove_internal_silence,
        top_db=top_db,
        seg_sec=seg_sec,
        hop_sec=hop_sec,
        metadata_csv_path=metadata_csv_path,
        long_files_csv_path=long_files_csv_path,
        long_audio_dir=long_audio_dir,
        no_segment_min_sec=no_segment_min_sec,
        return_df=return_df,
        resume=resume,
        save_every=save_every,
    )


# =========================================================
# Public API: Prepare MyST
# =========================================================
def prepare_myst_dataset(
    myst_root_dir: str,
    processed_dir: str,
    remove_internal_silence: bool = False,
    top_db: int = 30,
    seg_sec: float = 3.0,
    hop_sec: float = 1.5,
    splits: list[str] | None = None,
    metadata_csv_path: str | None = None,
    long_files_csv_path: str | None = None,
    long_audio_dir: str | None = None,
    no_segment_min_sec: float = 10.0,
    return_df: bool = True,
    resume: bool = True,
    save_every: int = 100,
) -> pd.DataFrame | None:
    """
    Prepare processed MyST data.

    Parameters
    ----------
    myst_root_dir : str
        Root directory of the MyST dataset.
    processed_dir : str
        Directory where processed segments will be saved.
    remove_internal_silence : bool, default=False
        Whether to remove internal non-speech regions.
    top_db : int, default=30
        Silence threshold used for edge trimming.
    seg_sec : float, default=3.0
        Segment length in seconds.
    hop_sec : float, default=1.5
        Hop length in seconds.
    splits : list[str] | None, default=None
        Dataset splits to process. Supported values are
        ["train", "valid", "test"].

        If None, all available splits are processed.
    metadata_csv_path : str | None, optional
        Output CSV path for segment-level metadata.
    long_files_csv_path : str | None, optional
        Output CSV path for long processed files metadata.
    long_audio_dir : str | None, optional
        External folder for long processed full audio files.
    no_segment_min_sec : float, default=10.0
        Duration threshold for skipping segmentation.
    return_df : bool, default=True
        Whether to return the segment-level dataframe.
    resume : bool, default=True
        Whether to continue from previously saved metadata.
    save_every : int, default=100
        Number of newly processed original audio files after which progress
        is written to disk.

    Returns
    -------
    pd.DataFrame | None
        Segment-level metadata dataframe for MyST.
    """
    df = load_myst_audio_paths(myst_root_dir, splits=splits)

    return process_segments(
        df=df,
        processed_dir=processed_dir,
        dataset_name="myst",
        spoof_label=0,
        remove_internal_silence=remove_internal_silence,
        top_db=top_db,
        seg_sec=seg_sec,
        hop_sec=hop_sec,
        metadata_csv_path=metadata_csv_path,
        long_files_csv_path=long_files_csv_path,
        long_audio_dir=long_audio_dir,
        no_segment_min_sec=no_segment_min_sec,
        return_df=return_df,
        resume=resume,
        save_every=save_every,
    )


# =========================================================
# Public API: Generic Dispatcher
# =========================================================
def prepare_dataset(
    dataset_name: str,
    processed_dir: str,
    remove_internal_silence: bool = False,
    top_db: int = 30,
    seg_sec: float = 3.0,
    hop_sec: float = 1.5,
    myst_root_dir: str | None = None,
    splits: list[str] | None = None,
    metadata_csv_path: str | None = None,
    long_files_csv_path: str | None = None,
    long_audio_dir: str | None = None,
    no_segment_min_sec: float = 10.0,
    return_df: bool = True,
    resume: bool = True,
    save_every: int = 100,
) -> pd.DataFrame | None:
    """
    Prepare one dataset by name.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier. Supported values are 'cv' and 'myst'.
    processed_dir : str
        Directory where processed segments will be saved.
    remove_internal_silence : bool, default=False
        Whether to remove internal non-speech regions.
    top_db : int, default=30
        Silence threshold used for edge trimming.
    seg_sec : float, default=3.0
        Segment length in seconds.
    hop_sec : float, default=1.5
        Hop length in seconds.
    myst_root_dir : str | None, default=None
        Root directory of MyST. Required only when dataset_name='myst'.
    splits : list[str] | None, default=None
        Optional list of dataset splits to process.
        This parameter is only used when dataset_name='myst'.
    metadata_csv_path : str | None, optional
        Output CSV path for segment-level metadata.
    long_files_csv_path : str | None, optional
        Output CSV path for long processed files metadata.
    long_audio_dir : str | None, optional
        External folder for long processed full audio files.
    no_segment_min_sec : float, default=10.0
        Duration threshold for skipping segmentation.
    return_df : bool, default=True
        Whether to return the segment-level dataframe.
    resume : bool, default=True
        Whether to continue from previously saved metadata.
    save_every : int, default=100
        Number of newly processed original audio files after which progress
        is written to disk.

    Returns
    -------
    pd.DataFrame | None
        Segment-level metadata dataframe.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "cv":
        return prepare_common_voice_dataset(
            processed_dir=processed_dir,
            remove_internal_silence=remove_internal_silence,
            top_db=top_db,
            seg_sec=seg_sec,
            hop_sec=hop_sec,
            metadata_csv_path=metadata_csv_path,
            long_files_csv_path=long_files_csv_path,
            long_audio_dir=long_audio_dir,
            no_segment_min_sec=no_segment_min_sec,
            return_df=return_df,
            resume=resume,
            save_every=save_every,
        )

    if dataset_name == "myst":
        if myst_root_dir is None:
            raise ValueError("myst_root_dir must be provided for MyST.")

        return prepare_myst_dataset(
            myst_root_dir=myst_root_dir,
            processed_dir=processed_dir,
            remove_internal_silence=remove_internal_silence,
            top_db=top_db,
            seg_sec=seg_sec,
            hop_sec=hop_sec,
            splits=splits,
            metadata_csv_path=metadata_csv_path,
            long_files_csv_path=long_files_csv_path,
            long_audio_dir=long_audio_dir,
            no_segment_min_sec=no_segment_min_sec,
            return_df=return_df,
            resume=resume,
            save_every=save_every,
        )

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
