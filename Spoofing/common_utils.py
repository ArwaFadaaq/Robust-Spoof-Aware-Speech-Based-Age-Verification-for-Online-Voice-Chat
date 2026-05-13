"""
Common Utilities for Spoofing Pipeline (Train / Valid / Test)

This module contains shared functions used by all spoofing pipelines:
    - Manifest schema and writer (with resume support)
    - Audio post-processing (VAD for TTS, padding/trimming to 3s @ 16kHz)
    - Safe value handling for manifest rows
    - Age-direction derivation
"""

import os
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch

# Import VAD from the preprocessing module of this repository
from preprocessing import run_silero_vad


# =========================================================
# Constants
# =========================================================

TARGET_SR        = 16000
TARGET_DURATION  = 3.0
TARGET_LEN       = int(TARGET_SR * TARGET_DURATION)


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
    "setting",
]


def create_empty_manifest(path):
    """Create an empty manifest CSV with the standard schema if missing."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(path, index=False)
    return path


class ManifestWriter:
    """Append-only manifest writer with periodic flushing."""

    def __init__(self, csv_path, flush_every=5):
        self.csv_path    = csv_path
        self.flush_every = flush_every
        self._buf        = []
        if not os.path.exists(csv_path):
            create_empty_manifest(csv_path)

    def append(self, row):
        self._buf.append({c: row.get(c, np.nan) for c in MANIFEST_COLUMNS})
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._buf:
            pd.DataFrame(self._buf, columns=MANIFEST_COLUMNS).to_csv(
                self.csv_path, mode="a", header=False, index=False, na_rep="NaN"
            )
            self._buf.clear()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


# =========================================================
# Safe value helpers
# =========================================================

def safe_val(v):
    """Replace None or NaN with np.nan."""
    if v is None:
        return np.nan
    if isinstance(v, float) and math.isnan(v):
        return np.nan
    return v


def safe_str(v):
    """Return a lowercase stripped string, or None for missing values."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return str(v).strip().lower()


def derive_age_direction(source_age, target_age):
    """Return short age direction code like 'a2m' or 'm2a'."""
    def _short(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        x = str(v).strip().lower()
        if x in ("minor", "m"):
            return "m"
        if x in ("adult", "a"):
            return "a"
        return None

    a, b = _short(source_age), _short(target_age)
    return f"{a}2{b}" if (a and b) else np.nan


# =========================================================
# Audio post-processing
# =========================================================

def _to_numpy(audio):
    """Convert any input (path, tensor, ndarray) to mono float32 numpy array."""
    if isinstance(audio, str) and os.path.exists(audio):
        a, _ = librosa.load(audio, sr=TARGET_SR)
    elif hasattr(audio, "detach"):
        a = audio.detach().cpu().numpy()
    else:
        a = np.asarray(audio)
    return a.squeeze().astype(np.float32)


def pad_or_trim_to_3s(audio):
    """
    Make any waveform exactly TARGET_LEN samples (3 seconds at 16 kHz).
    Longer audio is trimmed, shorter audio is zero-padded.
    """
    a = _to_numpy(audio)
    if len(a) > TARGET_LEN:
        a = a[:TARGET_LEN]
    elif len(a) < TARGET_LEN:
        a = np.pad(a, (0, TARGET_LEN - len(a)), mode="constant")
    return a


def apply_vad_longest_segment(audio):
    """
    Run Silero VAD on a waveform and return the longest detected
    speech segment as a numpy array at TARGET_SR.

    Returns
    -------
    np.ndarray or None
        The longest voiced region, or None if no speech was detected.
    """
    a = _to_numpy(audio)

    # Silero VAD expects a torch tensor
    waveform = torch.from_numpy(a).float()

    all_segs, long_segs, _ = run_silero_vad(
        waveform,
        sr=TARGET_SR,
        merge_gap_ms=500,
        min_speech_sec=0.5,
    )

    # Prefer long segments, fall back to all segments
    candidates = long_segs if long_segs else all_segs
    if not candidates:
        return None

    # Pick the longest segment
    longest = max(candidates, key=lambda se: se[1] - se[0])
    start_sec, end_sec = longest
    start_sample = int(start_sec * TARGET_SR)
    end_sample   = int(end_sec   * TARGET_SR)
    return a[start_sample:end_sample]


def save_padded(out_path, audio, apply_vad=False):
    """
    Save audio to disk as 16-bit PCM WAV, exactly 3 seconds at 16 kHz.

    Parameters
    ----------
    out_path : str
        Destination WAV path.
    audio : str | torch.Tensor | np.ndarray
        Input waveform (file path, tensor, or numpy array).
    apply_vad : bool
        If True, run VAD first and keep the longest speech segment
        before padding or trimming. Used for TTS outputs.

    Returns
    -------
    str
        The output path on success.

    Raises
    ------
    ValueError
        If apply_vad is True and no speech was detected.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if apply_vad:
        voiced = apply_vad_longest_segment(audio)
        if voiced is None or len(voiced) < int(0.2 * TARGET_SR):
            raise ValueError("VAD found no usable speech in TTS output")
        a = pad_or_trim_to_3s(voiced)
    else:
        a = pad_or_trim_to_3s(audio)

    sf.write(out_path, a, TARGET_SR, subtype="PCM_16")
    return out_path


# =========================================================
# Engine assignment helper (used by test_pipeline)
# =========================================================

def assign_engines_balanced(pool, engines, cross_age_p, rng,
                            flip_age_map, is_replay=False):
    """
    Distribute a pool of segments evenly across engines, then mark
    cross-age within each engine bucket using probability cross_age_p.

    Parameters
    ----------
    pool : list of dict
        Source segment rows.
    engines : list of str
        Engine names that should receive segments.
    cross_age_p : float
        Probability of marking a segment as cross-age (0.0 - 1.0).
    rng : np.random.Generator
    flip_age_map : dict
        Mapping like {"adult": "minor", "minor": "adult"}.
    is_replay : bool
        If True, all segments are marked cross_age = False and
        target_age_class is left as NaN.

    Returns
    -------
    list of dict
        Same rows with added keys: spoof_engine, cross_age_spoof,
        target_age_class.
    """
    buckets = {e: [] for e in engines}
    for i, r in enumerate(pool):
        r["spoof_engine"] = engines[i % len(engines)]
        buckets[r["spoof_engine"]].append(r)

    result = []
    for eng, bucket in buckets.items():
        n = len(bucket)

        if is_replay:
            for r in bucket:
                r["cross_age_spoof"]  = False
                r["target_age_class"] = np.nan
                result.append(r)
            continue

        cross_n = int(round(n * cross_age_p))
        idxs    = np.arange(n)
        rng.shuffle(idxs)
        cross_set = set(idxs[:cross_n].tolist())

        for i, r in enumerate(bucket):
            r["cross_age_spoof"]  = (i in cross_set)
            src_age               = r["mapped_age_class"]
            r["target_age_class"] = flip_age_map[src_age] if r["cross_age_spoof"] else src_age
            result.append(r)

    return result
