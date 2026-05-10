"""
Audio Preprocessing Module

This module implements the audio preprocessing pipeline for the
age verification and anti-spoofing graduation project (GP2).

Processing Pipeline
-------------------
1. Load
   Read the audio file from disk. Supports mp3, wav, flac, and m4a.

2. Normalize
   Convert to single-channel mono by averaging all channels.
   Resample to 16 kHz if the original sample rate differs.

3. Clipping detection
   Clipping is detected at the segment level.
   Segments with amplitude ≥ 0.999 are flagged in the segment manifest.

4. Voice Activity Detection (VAD)  [real_candidates only]
   The waveform is processed frame by frame using Silero VAD.
   Each 30 ms frame is classified as speech or non-speech.
   Consecutive speech frames are grouped into speech regions.
   Two adjacent regions separated by less than 500 ms of silence
   are merged into one continuous speech region.

   Only speech regions with duration >= 3 seconds are kept
   for segmentation.

5. Lightweight Silence Filtering  [spoof_targets only]
   Spoof target files are filtered using short RMS-energy windows
   instead of full VAD. The waveform is divided into fixed 50 ms
   windows and RMS energy is computed for each window separately.
   Files with a high ratio of low-energy windows are discarded.
   No segmentation is applied.

6. Segmentation  [real_candidates only]
   Each voiced region is split into fixed 3-second windows.
   The hop size depends on the split:
   - Train : 1.5-second hop (overlapping windows).
   - Val   : 3-second hop (no overlap).
   - Test  : 3-second hop (no overlap).

7. Storage
   The storage behavior depends on the processing mode:

   - real_candidates:
     Extracted segments are saved under:
     processed/real_candidates/<dataset_split>/<speaker_id>/

   - spoof_targets:
     The normalized full audio file is saved under:
     processed/spoof_targets/<dataset_split>/<speaker_id>/

8. Manifest generation
   Two types of CSV files are used for tracking:

   - file manifest:
     one row per processed file (used for both real and spoof)

   - segment manifest:
     one row per saved segment (used only for real_candidates)

Resume Support
--------------
If a file manifest CSV already exists, the pipeline loads it
and skips any file that has already been processed based on its file_id.

Checkpoint
----------
Manifest files are written to disk every N files (default 500)
to limit the amount of work lost if the session is interrupted.
"""

import os
import math

import torch
import torchaudio
import pandas as pd
from tqdm.notebook import tqdm
from silero_vad import load_silero_vad, get_speech_timestamps


# =========================================================
# Configuration
# =========================================================

# Load the Silero VAD model once during import
silero_model = load_silero_vad()

TARGET_SR      = 16000   # target sampling rate for all audio
WINDOW_SEC     = 3.0     # segment length in seconds
HOP_TRAIN_SEC  = 1.5     # hop size for train split (overlapping)
MIN_SPEECH_SEC = 3.0     # minimum continuous speech to keep a file
MERGE_GAP_MS   = 500     # max silence gap in ms to merge VAD segments
CLIP_THRESHOLD = 0.999   # amplitude threshold for clipping detection


# =========================================================
# Helpers
# =========================================================

def get_dataset_short(dataset: str) -> str:
    """
    Map dataset name to a short folder prefix.
    """
    mapping = {
        "common_voice": "cv",
        "myst": "myst",
        "voxceleb": "vox",
    }
    return mapping[dataset]


# =========================================================
# VAD helpers
# =========================================================

def run_silero_vad(
    waveform: torch.Tensor,
    sr: int = TARGET_SR,
    merge_gap_ms: int = MERGE_GAP_MS,
    min_speech_sec: float = MIN_SPEECH_SEC
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float]:
    """
    Run Silero VAD and return speech segments in seconds.

    Returns
    -------
    all_voiced_segs : list of (start_sec, end_sec)
        All detected speech segments after merging short silence gaps.
    long_voiced_segs : list of (start_sec, end_sec)
        Only speech segments whose duration is >= min_speech_sec.
    voiced_duration_sec : float
        Total duration of long_voiced_segs only.
    """
    waveform = waveform.float().cpu()

    speech_timestamps = get_speech_timestamps(
        waveform,
        silero_model,
        sampling_rate=sr
    )

    if len(speech_timestamps) == 0:
        return [], [], 0.0

    # convert sample indices to seconds
    segments_sec = [
        (item["start"] / sr, item["end"] / sr)
        for item in speech_timestamps
    ]

    # merge segments if silence gap between them is < merge_gap_ms
    merged = []
    max_gap_sec = merge_gap_ms / 1000.0

    for start_sec, end_sec in segments_sec:
        if merged and (start_sec - merged[-1][1]) < max_gap_sec:
            merged[-1][1] = end_sec
        else:
            merged.append([start_sec, end_sec])

    all_voiced_segs = [(s, e) for s, e in merged]

    # Keep only speech regions long enough for segmentation
    long_voiced_segs = [
        (s, e) for s, e in all_voiced_segs
        if (e - s) >= min_speech_sec
    ]

    voiced_duration_sec = sum(e - s for s, e in long_voiced_segs)

    return all_voiced_segs, long_voiced_segs, voiced_duration_sec


# =========================================================
# Audio helpers
# =========================================================

def is_clipped(waveform, threshold=CLIP_THRESHOLD):
    """
    Check whether a waveform contains clipped samples.

    Clipping occurs when the amplitude reaches its maximum value,
    which usually happens when a speaker is too close to the microphone.
    Clipped files are flagged in the manifest but not discarded.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform.
    threshold : float
        Amplitude threshold for clipping detection.

    Returns
    -------
    bool
        True if clipping is detected, False otherwise.
    """
    return bool((waveform.abs() >= threshold).any())


def segment_audio(waveform, sr, window_sec=WINDOW_SEC, hop_sec=None):
    """
    Split a waveform into fixed-length windows.

    For train splits, overlapping windows are used to increase the
    number of training samples without collecting more data.
    For val and test splits, non-overlapping windows are used to
    avoid evaluating the model on duplicate segments.

    Parameters
    ----------
    waveform : torch.Tensor
        Input waveform.
    sr : int
        Sampling rate.
    window_sec : float
        Window length in seconds.
    hop_sec : float or None
        Hop length in seconds. None means no overlap.

    Returns
    -------
    list of (float, float)
        List of (start_sec, end_sec) for each window.
    """
    hop     = hop_sec if hop_sec else window_sec
    win_len = int(window_sec * sr)
    hop_len = int(hop * sr)
    total   = waveform.shape[-1]
    segs    = []
    start   = 0
    while start + win_len <= total:
        segs.append((start / sr, (start + win_len) / sr))
        start += hop_len
    return segs


def load_segment_for_model(audio_path, target_sr=TARGET_SR):
    """
    Load a saved segment and prepare it as model input.

    Parameters
    ----------
    audio_path : str
        Path to the saved segment WAV file.
    target_sr : int
        Target sampling rate.

    Returns
    -------
    torch.Tensor
        Mono waveform ready for model input.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


# =========================================================
# Manifest row builders
# =========================================================

def _build_file_row(file_id, spk_id, dataset, split, pool, source_path,
                    processed_path, raw_dur, voiced_segs, voiced_dur, corrupted_flag):
    """Build a single row for file_manifest.csv."""
    return {
        "file_id":             file_id,
        "speaker_id":          spk_id,
        "dataset":             dataset,
        "split":               split,
        "pool":                pool,
        "source_path":         source_path,
        "processed_path":      processed_path,
        "raw_duration_sec":    round(raw_dur, 3),
        "voiced_segs_sec":     str(voiced_segs),
        "voiced_duration_sec": round(voiced_dur, 3),
        "is_corrupted":        corrupted_flag,
    }


def _build_seg_row(seg_id, file_id, start_sec, end_sec,spk_id,
                   age, dataset, split, pool, seg_path, clipped_flag):
    """Build a single row for segment_manifest.csv."""
    return {
        "segment_id":       seg_id,
        "parent_file_id":   file_id,
        "start_sec":        round(start_sec, 3),
        "end_sec":          round(end_sec, 3),
        "speaker_id":       spk_id,
        "mapped_age_class": age,
        "dataset_source":   dataset,
        "split":            split,
        "pool":             pool,
        "seg_path":         seg_path,
        "is_clipped":       clipped_flag,
    }


# =========================================================
# Single-file processing
# =========================================================

def process_file(row, file_id, hop_sec, processed_dir,
                 file_manifest_rows, seg_manifest_rows,
                 save_mode="segments"):
    """
    Process one audio file.

    Steps
    -----
    1. Load the audio file and convert it to mono at TARGET_SR.
    2. Check for clipping.
    3. If save_mode == "files", apply lightweight filtering
       for spoof targets:
       - discard files shorter than MIN_SPEECH_SEC
       - apply lightweight RMS-based silence filtering
       - discard files dominated by low-energy windows
       Accepted files are then saved as full audio.
    4. If save_mode == "segments", run Silero VAD and keep only
       voiced regions >= MIN_SPEECH_SEC.
    5. If save_mode == "segments", split each valid voiced region
       into fixed-length segments.
    6. Save the corresponding outputs and update the manifest rows.

    Parameters
    ----------
    row : pd.Series
        One row from file_list containing:
        path, speaker_id, dataset, split, pool.
    file_id : str
        Unique identifier for this file.
    hop_sec : float or None
        Hop size for segmentation. HOP_TRAIN_SEC for train, None for val/test.
    processed_dir : str
        Root directory for the current role output.
    file_manifest_rows : list
        Accumulator list for file-level manifest rows.
    seg_manifest_rows : list
        Accumulator list for segment-level manifest rows.
    save_mode : str
        "segments" for real_candidates, "files" for spoof_targets.

    Returns
    -------
    int
        Number of saved segments. Returns 0 for full-file mode or if
        no valid segments are created.
    """
    src_path = row["path"]
    spk_id   = row["speaker_id"]
    dataset  = row["dataset"]
    split    = row["split"]
    pool     = row["pool"]
    age      = row["pool"].split("_")[0]

    # Build speaker output directory under <dataset_split>/<speaker_id>
    dataset_split = f"{get_dataset_short(dataset)}_{split}"
    spk_dir = os.path.join(processed_dir, dataset_split, str(spk_id))
    os.makedirs(spk_dir, exist_ok=True)

    processed_path = ""

    corrupted_flag = False
    voiced_segs    = []
    voiced_dur     = 0.0
    raw_dur        = 0.0
    file_seg_count = 0

    try:
        waveform, sr = torchaudio.load(src_path)

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample to TARGET_SR
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        waveform     = waveform.squeeze(0)
        raw_dur      = waveform.shape[-1] / TARGET_SR

        if save_mode == "files":

            # Reject files shorter than 3 seconds
            if raw_dur < MIN_SPEECH_SEC:
                return "short"

            # Lightweight silence filtering using short RMS windows
            WINDOW_MS = 50
            ENERGY_THRESHOLD = 0.003
            MAX_SILENCE_RATIO = 0.80

            # window size in samples
            window_size = int(
                TARGET_SR * (WINDOW_MS / 1000.0)
            )

            # number of complete windows
            num_windows = waveform.numel() // window_size

            if num_windows == 0:
                return "low_speech"

            # reshape waveform into fixed-size windows
            trimmed = waveform[:num_windows * window_size]
            windows = trimmed.reshape(num_windows, window_size)

            # compute RMS energy per window
            window_rms = torch.sqrt(
                torch.mean(windows.float() ** 2, dim=1)
            )

            # ratio of low-energy windows
            silence_ratio = (
                (window_rms < ENERGY_THRESHOLD)
                .sum()
                .item()
                / num_windows
            )

            # reject files dominated by silence
            if silence_ratio >= MAX_SILENCE_RATIO:
                return "low_speech"

            # Save accepted spoof target
            processed_path = os.path.join(spk_dir, f"{file_id}.wav")

            torchaudio.save(
                processed_path,
                waveform.unsqueeze(0),
                TARGET_SR
            )

            file_manifest_rows.append(_build_file_row(
                file_id, spk_id, dataset, split, pool,
                src_path, processed_path, raw_dur,
                [], raw_dur * (1.0 - silence_ratio),
                corrupted_flag
            ))

            return 0

        # run Silero  VAD — keep only regions >= MIN_SPEECH_SEC
        voiced_segs, long_segs, voiced_dur = run_silero_vad(
            waveform,
            sr=TARGET_SR,
            merge_gap_ms=MERGE_GAP_MS,
            min_speech_sec=MIN_SPEECH_SEC
        )

        # discard file if no qualifying speech found
        if not long_segs:
            file_manifest_rows.append(_build_file_row(
                file_id, spk_id, dataset, split, pool,
                src_path, "", raw_dur, voiced_segs,
                voiced_dur, corrupted_flag
            ))
            return 0

        # segment each voiced region and save
        for region_idx, (vs, ve) in enumerate(long_segs):
            chunk    = waveform[int(vs * TARGET_SR): int(ve * TARGET_SR)]
            seg_list = segment_audio(chunk, TARGET_SR, WINDOW_SEC, hop_sec)

            for seg_idx, (ss, se) in enumerate(seg_list):
                seg_id = f"{file_id}_r{region_idx:02d}_seg{seg_idx:04d}"
                seg_audio = chunk[int(ss * TARGET_SR): int(se * TARGET_SR)]
                seg_clipped = is_clipped(seg_audio)
                seg_path = os.path.join(spk_dir, f"{seg_id}.wav")
                torchaudio.save(seg_path, seg_audio.unsqueeze(0), TARGET_SR)
                file_seg_count += 1

                seg_manifest_rows.append(_build_seg_row(
                    seg_id, file_id,
                    ss + vs, se + vs,
                    spk_id, age, dataset,
                    split, pool, seg_path, seg_clipped
                ))

    except Exception as ex:
        corrupted_flag = True
        print(f"  CORRUPTED: {src_path} — {ex}")

    file_manifest_rows.append(_build_file_row(
        file_id, spk_id, dataset, split, pool,
        src_path, "", raw_dur, voiced_segs,
        voiced_dur, corrupted_flag
    ))

    return file_seg_count


# =========================================================
# Pipeline
# =========================================================

def run_pipeline(file_list, dataset, split, processed_dir, manifest_dir,
                 hop_sec, checkpoint_every=500, save_mode="segments"):
    """
    Run the preprocessing pipeline for the given splits.

    Loads existing manifests if available to support resume.
    Saves manifests every checkpoint_every files to prevent data loss.

    Parameters
    ----------
    file_list : pd.DataFrame
        Full file list from data_preparation.build_file_list().
    dataset : str
        Dataset name (e.g., "common_voice", "myst", "voxceleb").
    split : str
        Data split (e.g., "train", "val", "test").
    processed_dir : str
        Root directory for the current role output.
    manifest_dir : str
        Directory where manifest CSV files are saved.
    hop_sec : float or None
        Hop size for segmentation. HOP_TRAIN_SEC for train, None for val/test.
    checkpoint_every : int
        Number of files after which manifests are saved to disk.
    save_mode : str
        "segments" for real_candidates, "files" for spoof_targets.
    """

    os.makedirs(manifest_dir, exist_ok=True)

    dataset_split = f"{get_dataset_short(dataset)}_{split}"

    file_manifest_path = os.path.join(manifest_dir, f"file_manifest_{dataset_split}.csv")

    if save_mode == "segments":
        seg_manifest_path = os.path.join(manifest_dir, f"segment_manifest_{dataset_split}.csv")
    else:
        seg_manifest_path = None

    # resume: load existing manifests if available
    if os.path.exists(file_manifest_path):
        done_df = pd.read_csv(file_manifest_path)
        done_ids = set(done_df["file_id"])
        file_manifest_rows = done_df.to_dict("records")

        if seg_manifest_path is not None and os.path.exists(seg_manifest_path):
            seg_manifest_rows = pd.read_csv(seg_manifest_path).to_dict("records")
        else:
            seg_manifest_rows = []

        print(f"Resuming — {len(done_ids)} files done, {len(seg_manifest_rows)} segments found")
    else:
        done_ids = set()
        file_manifest_rows = []
        seg_manifest_rows = []
        print("Starting fresh")

    # filter target files and remove already-processed ones
    target = file_list[
        (file_list["dataset"] == dataset) &
        (file_list["split"] == split)
    ].copy()

    if target.empty:
        print("No files to process.")
        return

    target = target[~target["file_id"].isin(done_ids)].copy()

    print(f"Files remaining: {len(target)}")

    if target.empty:
        print(f"{dataset_split} already processed.")
        return

    seg_count = len(seg_manifest_rows)
    skipped_short = 0
    skipped_low_speech = 0

    pbar = tqdm(
        target.iterrows(),
        total=len(target),
        desc=dataset_split,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files | segs: {postfix} | {elapsed}<{remaining}",
    )

    for _, row in pbar:
        file_id = row["file_id"]

        result = process_file(
            row, file_id, hop_sec,
            processed_dir,
            file_manifest_rows,
            seg_manifest_rows,
            save_mode=save_mode
        )

        if result == "short":
            skipped_short += 1

        elif result == "low_speech":
            skipped_low_speech += 1

        else:
            seg_count += result

        pbar.set_postfix_str(f"{seg_count}")

        # checkpoint every N files
        if len(file_manifest_rows) % checkpoint_every == 0:
            pd.DataFrame(file_manifest_rows).to_csv(file_manifest_path, index=False)

            if seg_manifest_path is not None:
                pd.DataFrame(seg_manifest_rows).to_csv(seg_manifest_path, index=False)

    # final save
    pd.DataFrame(file_manifest_rows).to_csv(file_manifest_path, index=False)

    if seg_manifest_path is not None:
        pd.DataFrame(seg_manifest_rows).to_csv(seg_manifest_path, index=False)

    print(f"\n{dataset_split} done.")
    print(f"Files:    {len(file_manifest_rows)}")

    if save_mode == "files":
        print(f"Skipped (<3s): {skipped_short}")
        print(f"Skipped (low speech): {skipped_low_speech}")

    if seg_manifest_path is not None:
        print(f"Segments: {len(seg_manifest_rows)}")
