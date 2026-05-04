# -*- coding: utf-8 -*-
"""
Spoofing/spoof_utils.py
=======================

Manifest utilities for the spoofing pipeline.

Drop this file into the `Spoofing/` folder of the repo, then in your spoofing
notebook:

    from Spoofing.spoof_utils import (
        create_empty_manifest,
        ManifestWriter,
    )

Workflow per run:
    1. Call `create_empty_manifest('.../my_manifest.csv')` once to make the CSV.
    2. Open a writer:   `mw = ManifestWriter('.../my_manifest.csv', flush_every=25)`
    3. For each generated spoof clip, call ONE of:
         - mw.append_vc(source_row, target_row, seg_path, engine, cross_age_spoof, **extras)
         - mw.append_tts(source_row, target_row, seg_path, engine, transcript_id, cross_age_spoof, **extras)
         - mw.append_replay(source_row, seg_path, engine, **extras)
    4. At the end, call `mw.close()` to flush the last partial buffer.

Every 25 rows the writer auto-flushes to Drive. If Colab disconnects you lose
at most 24 rows from the in-memory buffer (the rest are safe on disk).

Replay rows leave the target / transcript columns as NaN — the schema stays
consistent so downstream code never has to special-case attack types.
"""

from __future__ import annotations

import os
import math
import hashlib
from typing import Any, Mapping, Optional, Iterable

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Canonical schema — keep this list in sync with whatever the rest of
# the project expects.
# ------------------------------------------------------------------ #
MANIFEST_COLUMNS = [
    # ===== IDs =====
    "segment_id",
    "parent_file_id",
    # ===== Time =====
    "start_sec",
    "end_sec",
    # ===== Paths =====
    "seg_path",
    "source_file_path",
    "target_file_path",
    # ===== Source Info =====
    "source_speaker_id",
    "source_gender",
    "source_dataset",
    "source_age_class",
    # ===== Target Info (spoof only) =====
    "target_speaker_id",
    "target_gender",
    "target_age_class",
    # ===== Labels =====
    "authenticity",        # real | spoof
    "spoof_type",          # none | vc | tts | replay
    "spoof_engine",
    # ===== Spoof Logic =====
    "cross_age_spoof",
    "age_direction",       # m2m, a2a, m2a, a2m
    "source_transcript_id",
    # ===== Dataset Info =====
    "dataset_source",
    "split",
    "pool",
    # ===== Original =====
    "is_clipped",
]


# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #
def create_empty_manifest(output_csv: str) -> str:
    """
    Create an empty spoofing manifest with ALL required columns and save it
    to `output_csv`. Returns the path.

    Safe to re-run: if the file already exists it is left untouched.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    if os.path.exists(output_csv):
        # Verify the schema, but don't overwrite.
        try:
            existing = pd.read_csv(output_csv, nrows=0)
            if list(existing.columns) != MANIFEST_COLUMNS:
                print(f"[spoof_utils] WARNING: {output_csv} exists but has a "
                      f"different schema. Leaving it unchanged.")
        except Exception:
            pass
        print(f"[spoof_utils] manifest already exists: {output_csv}")
        return output_csv

    df = pd.DataFrame(columns=MANIFEST_COLUMNS)
    df.to_csv(output_csv, index=False)
    print(f"Empty manifest created: {output_csv}")
    print(f"Columns: {len(MANIFEST_COLUMNS)}")
    return output_csv


def derive_age_direction(source_age_class: Any,
                         target_age_class: Optional[Any]) -> Any:
    """
    Map (source_age_class, target_age_class) → one of m2m / a2a / m2a / a2m.
    Returns NaN if either side is missing/unknown (replay attacks pass None
    for the target, so they always get NaN here).
    """
    def _short(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        s = str(v).strip().lower()
        if s in ("minor", "minors", "kid", "kids", "child", "children", "m"):
            return "m"
        if s in ("adult", "adults", "a"):
            return "a"
        return None

    s = _short(source_age_class)
    t = _short(target_age_class)
    if s is None or t is None:
        return np.nan
    return f"{s}2{t}"


def compute_is_clipped(audio: Any, threshold: float = 0.99) -> Optional[bool]:
    """
    Return True if the audio reaches/saturates the int16 range. Accepts a
    file path, a numpy array, or a torch tensor. Returns None if it can't
    figure out the input.
    """
    try:
        if audio is None:
            return None
        if isinstance(audio, str):
            if not os.path.exists(audio):
                return None
            import soundfile as sf
            data, _ = sf.read(audio, always_2d=False)
            arr = np.asarray(data)
        elif hasattr(audio, "detach"):
            arr = audio.detach().cpu().numpy()
        else:
            arr = np.asarray(audio)
        if arr.size == 0:
            return None
        # Normalize to [-1, 1] if int16
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        return bool(np.max(np.abs(arr)) >= threshold)
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Internal: pull values from a source/target row tolerantly
# ------------------------------------------------------------------ #
_PATH_COL_CANDIDATES = (
    "file_path", "wav_path", "path", "filepath", "filename",
    "audio_path", "segment_path", "seg_path",
)
_AGE_CLASS_CANDIDATES = (
    "mapped_age_class", "age_class", "Age_class", "age_category", "age_group",
)


def _pick(row: Mapping, *keys, default=np.nan):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def _row_path(row: Mapping) -> Any:
    return _pick(row, *_PATH_COL_CANDIDATES, default=np.nan)


def _row_age_class(row: Mapping) -> Any:
    val = _pick(row, *_AGE_CLASS_CANDIDATES, default=np.nan)
    if isinstance(val, str):
        return val.strip().lower()
    return val


def _make_segment_id(seg_path: str) -> str:
    """Stable id derived from the seg_path basename."""
    name = os.path.splitext(os.path.basename(seg_path))[0]
    if not name:
        name = hashlib.md5(seg_path.encode("utf-8")).hexdigest()[:16]
    return name


# ------------------------------------------------------------------ #
# Buffered writer
# ------------------------------------------------------------------ #
class ManifestWriter:
    """
    Append rows to a manifest CSV in batches.

    The writer keeps an in-memory buffer of pending rows and flushes them to
    disk every `flush_every` appends (default 25). Use `close()` at the end
    of the run to flush whatever is left.

    All append methods accept arbitrary `**extras` to override any column
    explicitly (handy when the source row uses a non-standard column name).
    """

    def __init__(self, csv_path: str, flush_every: int = 25,
                 default_pool: str = "spoof_source"):
        self.csv_path = csv_path
        self.flush_every = max(1, int(flush_every))
        self.default_pool = default_pool
        self._buffer: list[dict] = []
        self._total_written = 0

        # Make sure the file exists with the right schema.
        if not os.path.exists(csv_path):
            create_empty_manifest(csv_path)

    # ---------------- public API ----------------

    def append_vc(self,
                  source_row: Mapping,
                  target_row: Mapping,
                  seg_path: str,
                  engine: str,
                  cross_age_spoof: bool,
                  **extras) -> None:
        """Append one VC spoof row."""
        self._append(
            self._build_row(
                source_row=source_row,
                target_row=target_row,
                seg_path=seg_path,
                spoof_type="vc",
                engine=engine,
                cross_age_spoof=cross_age_spoof,
                transcript_id=np.nan,
                extras=extras,
            )
        )

    def append_tts(self,
                   source_row: Mapping,
                   target_row: Mapping,
                   seg_path: str,
                   engine: str,
                   transcript_id: Any,
                   cross_age_spoof: bool,
                   **extras) -> None:
        """Append one TTS spoof row."""
        self._append(
            self._build_row(
                source_row=source_row,
                target_row=target_row,
                seg_path=seg_path,
                spoof_type="tts",
                engine=engine,
                cross_age_spoof=cross_age_spoof,
                transcript_id=transcript_id,
                extras=extras,
            )
        )

    def append_replay(self,
                      source_row: Mapping,
                      seg_path: str,
                      engine: str,
                      **extras) -> None:
        """
        Append one replay-attack spoof row.

        Replay has no target speaker and no transcript, so all target_* columns
        plus source_transcript_id stay NaN. cross_age_spoof is always False
        and age_direction is NaN.
        """
        self._append(
            self._build_row(
                source_row=source_row,
                target_row=None,                 # signals 'no target'
                seg_path=seg_path,
                spoof_type="replay",
                engine=engine,
                cross_age_spoof=False,
                transcript_id=np.nan,
                extras=extras,
            )
        )

    def flush(self) -> int:
        """Write any buffered rows to disk now. Returns the number flushed."""
        if not self._buffer:
            return 0
        n = len(self._buffer)
        df_new = pd.DataFrame(self._buffer, columns=MANIFEST_COLUMNS)
        # Append mode: skip header if the file already has one.
        write_header = not (
            os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0
        )
        df_new.to_csv(self.csv_path, mode="a", header=write_header, index=False)
        self._total_written += n
        self._buffer.clear()
        return n

    def close(self) -> int:
        """Final flush. Always call this when you're done."""
        return self.flush()

    @property
    def total_written(self) -> int:
        return self._total_written

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    # ---------------- context manager sugar ----------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't swallow exceptions

    # ---------------- internals ----------------

    def _append(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def _build_row(self, *,
                   source_row: Mapping,
                   target_row: Optional[Mapping],
                   seg_path: str,
                   spoof_type: str,
                   engine: str,
                   cross_age_spoof: bool,
                   transcript_id: Any,
                   extras: Mapping) -> dict:
        """Pull values out of source/target rows + apply extras overrides."""
        src = source_row or {}
        tgt = target_row if target_row is not None else {}

        source_age = _row_age_class(src)
        target_age = _row_age_class(tgt) if target_row is not None else np.nan

        row = {
            # IDs
            "segment_id":          _pick(src, "segment_id", default=_make_segment_id(seg_path)),
            "parent_file_id":      _pick(src, "parent_file_id", "file_id"),
            # Time
            "start_sec":           _pick(src, "start_sec", "start"),
            "end_sec":             _pick(src, "end_sec", "end"),
            # Paths
            "seg_path":            seg_path,
            "source_file_path":    _row_path(src),
            "target_file_path":    _row_path(tgt) if target_row is not None else np.nan,
            # Source info
            "source_speaker_id":   _pick(src, "speaker_id", "source_speaker_id"),
            "source_gender":       _pick(src, "gender", "source_gender"),
            "source_dataset":      _pick(src, "dataset_source", "source_dataset"),
            "source_age_class":    source_age,
            # Target info
            "target_speaker_id":   _pick(tgt, "speaker_id", "target_speaker_id") if target_row is not None else np.nan,
            "target_gender":       _pick(tgt, "gender", "target_gender") if target_row is not None else np.nan,
            "target_age_class":    target_age,
            # Labels
            "authenticity":        "spoof",
            "spoof_type":          spoof_type,
            "spoof_engine":        engine,
            # Spoof logic
            "cross_age_spoof":     bool(cross_age_spoof) if target_row is not None else False,
            "age_direction":       derive_age_direction(source_age, target_age) if target_row is not None else np.nan,
            "source_transcript_id": transcript_id,
            # Dataset info
            "dataset_source":      _pick(src, "dataset_source"),
            "split":               _pick(src, "split"),
            "pool":                _pick(src, "pool", default=self.default_pool),
            # Original
            "is_clipped":          _pick(src, "is_clipped", default=np.nan),
        }

        # Apply any caller-supplied overrides last (lets you fix odd cases).
        unknown = [k for k in extras.keys() if k not in MANIFEST_COLUMNS]
        if unknown:
            raise KeyError(f"Unknown manifest columns in extras: {unknown}. "
                           f"Allowed: {MANIFEST_COLUMNS}")
        row.update(extras)
        return row


# ------------------------------------------------------------------ #
# Convenience: a one-call helper for callers who don't want to manage a writer
# ------------------------------------------------------------------ #
def append_real_row(csv_path: str,
                    source_row: Mapping,
                    seg_path: str,
                    **extras) -> None:
    """
    Append a single REAL (bonafide) row. No buffering — opens, appends, closes.
    Use this for the real_only manifests.
    """
    src = source_row or {}
    row = {
        "segment_id":          _pick(src, "segment_id", default=_make_segment_id(seg_path)),
        "parent_file_id":      _pick(src, "parent_file_id", "file_id"),
        "start_sec":           _pick(src, "start_sec", "start"),
        "end_sec":             _pick(src, "end_sec", "end"),
        "seg_path":            seg_path,
        "source_file_path":    _row_path(src),
        "target_file_path":    np.nan,
        "source_speaker_id":   _pick(src, "speaker_id", "source_speaker_id"),
        "source_gender":       _pick(src, "gender", "source_gender"),
        "source_dataset":      _pick(src, "dataset_source", "source_dataset"),
        "source_age_class":    _row_age_class(src),
        "target_speaker_id":   np.nan,
        "target_gender":       np.nan,
        "target_age_class":    np.nan,
        "authenticity":        "real",
        "spoof_type":          "none",
        "spoof_engine":        np.nan,
        "cross_age_spoof":     False,
        "age_direction":       np.nan,
        "source_transcript_id": np.nan,
        "dataset_source":      _pick(src, "dataset_source"),
        "split":               _pick(src, "split"),
        "pool":                _pick(src, "pool", default="real_only"),
        "is_clipped":          _pick(src, "is_clipped", default=np.nan),
    }
    row.update(extras)
    if not os.path.exists(csv_path):
        create_empty_manifest(csv_path)
    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(
        csv_path, mode="a", header=False, index=False
    )
