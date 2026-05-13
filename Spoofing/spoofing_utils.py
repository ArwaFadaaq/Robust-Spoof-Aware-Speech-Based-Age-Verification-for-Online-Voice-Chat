"""
Common Utilities for Spoofing Pipeline (Train / Valid / Test)

This module contains ONLY shared utilities:
- Manifest schema + writer
- Safe helpers
- Audio post-processing (padding/trimming + VAD import)
- Age-direction logic
- File naming helpers

No split-specific or test-specific logic is included here.
"""

# =========================================================
# Imports
# =========================================================

import os
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

from preprocessing import run_silero_vad, apply_vad_longest_segment


# =========================================================
# Constants
# =========================================================

TARGET_SR = 16000
TARGET_DURATION = 3.0
TARGET_LEN = int(TARGET_SR * TARGET_DURATION)


# =========================================================
# Manifest schema
# =========================================================

MANIFEST_COLUMNS = [
    "source_seg_id",
    "parent_file_id",
    "target_file_id",
    "start_sec",
    "end_sec",
    "source_seg_path",
    "source_file_path",
    "target_file_path",
    "source_speaker_id",
    "source_gender",
    "source_dataset",
    "target_dataset",
    "source_age_class",
    "target_speaker_id",
    "target_gender",
    "target_age_class",
    "authenticity",
    "spoof_type",
    "spoof_engine",
    "cross_age_spoof",
    "age_direction",
    "source_transcript_id",
    "source_transcript",
    "split",
    "source_pool",
    "target_pool",
    "final_seg_path",
]


# =========================================================
# Manifest helpers
# =========================================================

def create_empty_manifest(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(path, index=False)
    return path


class ManifestWriter:
    def __init__(self, csv_path, flush_every=5):
        self.csv_path = csv_path
        self.flush_every = flush_every
        self.buffer = []

        if not os.path.exists(csv_path):
            create_empty_manifest(csv_path)

    def append(self, row):
        self.buffer.append({c: row.get(c, np.nan) for c in MANIFEST_COLUMNS})
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buffer:
            pd.DataFrame(self.buffer, columns=MANIFEST_COLUMNS).to_csv(
                self.csv_path,
                mode="a",
                header=False,
                index=False,
                na_rep="NaN"
            )
            self.buffer = []

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =========================================================
# Safe helpers
# =========================================================

def safe_val(v):
    if v is None:
        return np.nan
    if isinstance(v, float) and math.isnan(v):
        return np.nan
    return v


def safe_str(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v).strip().lower()


def derive_age_direction(source_age, target_age):
    def _norm(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        v = str(v).strip().lower()
        if v in ("minor", "m"):
            return "m"
        if v in ("adult", "a"):
            return "a"
        return None

    a, b = _norm(source_age), _norm(target_age)
    return f"{a}2{b}" if (a and b) else np.nan


# =========================================================
# Audio processing (shared only)
# =========================================================

def _to_numpy(audio):
    if isinstance(audio, str) and os.path.exists(audio):
        a, _ = librosa.load(audio, sr=TARGET_SR)
    elif hasattr(audio, "detach"):
        a = audio.detach().cpu().numpy()
    else:
        a = np.asarray(audio)

    return a.squeeze().astype(np.float32)


def pad_or_trim_to_3s(audio):
    a = _to_numpy(audio)

    if len(a) > TARGET_LEN:
        a = a[:TARGET_LEN]
    elif len(a) < TARGET_LEN:
        a = np.pad(a, (0, TARGET_LEN - len(a)), mode="constant")

    return a


def save_padded(out_path, audio, apply_vad=False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if apply_vad:
        audio = apply_vad_longest_segment(audio)
        if audio is None or len(audio) < int(0.2 * TARGET_SR):
            raise ValueError("No valid speech detected by VAD")

    audio = pad_or_trim_to_3s(audio)

    sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
    return out_path


# =========================================================
# Manifest row builder (shared)
# =========================================================

def row_to_manifest(row, kind, final_seg_path, transcript_lookup):
    is_replay = (kind == "replay")

    src_age = safe_str(row.get("mapped_age_class"))
    tgt_age = np.nan if is_replay else safe_str(row.get("target_age_class"))

    return {
        "source_seg_id": safe_val(row.get("segment_id")),
        "parent_file_id": safe_val(row.get("parent_file_id")),
        "target_file_id": np.nan if is_replay else safe_val(row.get("target_file_id")),
        "start_sec": safe_val(row.get("start_sec")),
        "end_sec": safe_val(row.get("end_sec")),
        "source_seg_path": safe_val(row.get("seg_path")),
        "source_file_path": safe_val(row.get("source_file_path")),
        "target_file_path": np.nan if is_replay else safe_val(row.get("target_file_path")),
        "source_speaker_id": safe_val(row.get("speaker_id")),
        "source_gender": safe_val(row.get("gender")),
        "source_dataset": safe_val(row.get("dataset_source")),
        "target_dataset": np.nan if is_replay else safe_val(row.get("target_dataset")),
        "source_age_class": src_age,
        "target_speaker_id": np.nan if is_replay else safe_val(row.get("target_speaker_id")),
        "target_gender": np.nan if is_replay else safe_val(row.get("target_gender")),
        "target_age_class": tgt_age,
        "authenticity": "spoof",
        "spoof_type": kind,
        "spoof_engine": row["spoof_engine"],
        "cross_age_spoof": False if is_replay else bool(row.get("cross_age_spoof", False)),
        "age_direction": derive_age_direction(src_age, tgt_age) if not is_replay else np.nan,
        "source_transcript_id": safe_val(row.get("parent_file_id")) if kind == "tts" else np.nan,
        "source_transcript": (
            safe_val(transcript_lookup.get(str(row.get("segment_id", ""))))
            if kind == "tts" else np.nan
        ),
        "split": safe_val(row.get("split")),
        "source_pool": safe_val(row.get("pool")),
        "target_pool": np.nan if is_replay else safe_val(row.get("target_pool")),
        "final_seg_path": final_seg_path,
    }


# =========================================================
# Filename helper
# =========================================================

def make_filename(row, kind):
    return f"{kind}__{row['spoof_engine']}__{row['segment_id']}.wav"
