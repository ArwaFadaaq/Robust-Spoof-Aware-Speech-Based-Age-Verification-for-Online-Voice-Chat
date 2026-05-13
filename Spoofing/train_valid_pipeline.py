"""
Train / Valid Spoofing Pipeline (Dynamic, Resumable)

This module implements the dynamic source-to-engine assignment used
for the train and valid spoofed datasets.

Key behaviors
-------------
1. Source segments are split equally into three spoof types:
   one third for VC, one third for TTS, one third for Replay.

2. Within each type, segments are distributed equally across all
   engines belonging to that type.

3. For each engine, 80 percent of its segments are cross-age
   (the target age is the opposite of the source age).

4. Target speakers are picked with the following rules:
       - The speaker must satisfy the requested target age.
       - The speaker must have at least one audio file
         with duration >= MIN_TARGET_DURATION seconds.
       - If a speaker is rejected, another speaker is tried.

5. Engine failure handling:
       - If an engine crashes on a segment, the segment is reassigned
         to another engine of the same type, and the failing engine
         picks a different segment.
       - If all engines of the same type fail on a segment, the
         segment is logged to a failed CSV file.

6. Resume support:
       - A progress CSV is maintained, listing every
         (source_seg_id, spoof_engine) pair already completed.
       - On re-run, completed pairs are loaded and skipped.
"""

import os
import math
import gc
import traceback
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .common_utils import (
    ManifestWriter,
    create_empty_manifest,
    save_padded,
    derive_age_direction,
    safe_val,
    safe_str,
    TARGET_SR,
    TARGET_DURATION,
)


# =========================================================
# Constants
# =========================================================

FLIP_AGE = {"adult": "minor", "minor": "adult"}

MIN_TARGET_DURATION   = 7.0
MAX_TARGET_RETRIES    = 10
CROSS_AGE_DEFAULT     = 0.8


# =========================================================
# Progress tracker
# =========================================================

class ProgressTracker:
    """
    Tracks which (segment_id, engine) pairs are completed and which
    segment-engine pairs failed, so that a re-run does not redo work.
    """

    PROGRESS_COLUMNS = ["segment_id", "spoof_engine", "final_seg_path"]
    FAILED_COLUMNS   = ["segment_id", "spoof_engine", "error", "spoof_type"]

    def __init__(self, progress_csv, failed_csv):
        self.progress_csv = progress_csv
        self.failed_csv   = failed_csv

        os.makedirs(os.path.dirname(progress_csv) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(failed_csv)   or ".", exist_ok=True)

        if not os.path.exists(progress_csv):
            pd.DataFrame(columns=self.PROGRESS_COLUMNS).to_csv(progress_csv, index=False)
        if not os.path.exists(failed_csv):
            pd.DataFrame(columns=self.FAILED_COLUMNS).to_csv(failed_csv, index=False)

        self._done       = set()
        self._engine_bad = {}
        self._load()

    def _load(self):
        try:
            prog = pd.read_csv(self.progress_csv)
            for _, r in prog.iterrows():
                seg = str(r["segment_id"])
                eng = str(r["spoof_engine"])
                self._done.add((seg, eng))
        except Exception:
            pass

        try:
            fail = pd.read_csv(self.failed_csv)
            for _, r in fail.iterrows():
                seg = str(r["segment_id"])
                eng = str(r["spoof_engine"])
                self._engine_bad.setdefault(seg, set()).add(eng)
        except Exception:
            pass

    def is_done(self, segment_id, engine):
        return (str(segment_id), str(engine)) in self._done

    def engine_already_failed(self, segment_id, engine):
        return str(engine) in self._engine_bad.get(str(segment_id), set())

    def mark_done(self, segment_id, engine, final_path):
        key = (str(segment_id), str(engine))
        if key in self._done:
            return
        self._done.add(key)
        pd.DataFrame(
            [[str(segment_id), str(engine), final_path]],
            columns=self.PROGRESS_COLUMNS,
        ).to_csv(self.progress_csv, mode="a", header=False, index=False)

    def mark_engine_failed(self, segment_id, engine, error_msg, spoof_type):
        self._engine_bad.setdefault(str(segment_id), set()).add(str(engine))
        pd.DataFrame(
            [[str(segment_id), str(engine), str(error_msg)[:300], spoof_type]],
            columns=self.FAILED_COLUMNS,
        ).to_csv(self.failed_csv, mode="a", header=False, index=False)

    def segment_completed_for_type(self, segment_id, type_engines):
        """True if this segment has been completed by any engine of this type."""
        seg = str(segment_id)
        return any((seg, eng) in self._done for eng in type_engines)


# =========================================================
# Source pool builder
# =========================================================

def build_source_pools(source_df, valid_seg_ids, rng):
    """
    Split the source dataframe into three pools (vc / tts / replay).

    TTS requires a valid transcript, so segments without a transcript
    are kept out of the TTS pool.

    Returns three lists of dict rows.
    """
    shuffled = source_df.sample(
        frac=1, random_state=int(rng.integers(1e9))
    ).reset_index(drop=True)

    total       = len(shuffled)
    vc_target   = total // 3
    tts_target  = total // 3
    rep_target  = total - vc_target - tts_target

    vc_pool, tts_pool, rep_pool = [], [], []

    for _, row in shuffled.iterrows():
        r      = row.to_dict()
        seg_id = str(r["segment_id"])
        has_tr = seg_id in valid_seg_ids

        if len(vc_pool) < vc_target:
            vc_pool.append(r)
        elif len(tts_pool) < tts_target and has_tr:
            tts_pool.append(r)
        else:
            rep_pool.append(r)

    # If TTS is short, refill from replay pool using only rows with transcripts
    i = 0
    while len(tts_pool) < tts_target and i < len(rep_pool):
        if str(rep_pool[i]["segment_id"]) in valid_seg_ids:
            tts_pool.append(rep_pool.pop(i))
        else:
            i += 1

    vc_pool  = vc_pool[:vc_target]
    tts_pool = tts_pool[:tts_target]
    rep_pool = rep_pool[:rep_target]

    return vc_pool, tts_pool, rep_pool


# =========================================================
# Engine assignment within a type
# =========================================================

def assign_engines_and_cross_age(pool, engines, cross_age_p, rng):
    """
    Distribute a pool of segments evenly across engines, and mark
    cross-age within each engine bucket.

    Returns a list of dicts with added keys:
        spoof_engine, cross_age_spoof, target_age_class
    Replay segments always have cross_age_spoof = False.
    """
    is_replay = engines[0].startswith("replay_") if engines else False

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
            r["target_age_class"] = FLIP_AGE[src_age] if r["cross_age_spoof"] else src_age
            result.append(r)

    return result


# =========================================================
# Target speaker / file selection
# =========================================================

def pick_target(target_df, target_age, source_speaker_id, rng,
                rejected_speakers=None,
                min_duration=MIN_TARGET_DURATION,
                max_retries=MAX_TARGET_RETRIES):
    """
    Pick one target file matching the requested age class.

    Rules
    -----
    1. Filter targets by mapped_age_class == target_age.
    2. Pick a random speaker different from source_speaker_id
       and not in rejected_speakers.
    3. From that speaker's files, pick one with raw_duration_sec >= min_duration.
    4. If no qualifying file exists, reject the speaker and try another.

    Parameters
    ----------
    target_df : pd.DataFrame
        Spoof targets manifest. Must contain columns:
        speaker_id, mapped_age_class, raw_duration_sec, processed_path.
    target_age : str
        Required age class ("adult" or "minor"). May be NaN for replay
        (in which case this function should not be called).
    source_speaker_id : str
        The source speaker, which must not equal the picked target.
    rng : np.random.Generator
    rejected_speakers : set of str
        Speakers already rejected during this attempt cycle. Will be
        modified in place when speakers are rejected here.
    min_duration : float
        Minimum required file duration in seconds.
    max_retries : int
        Maximum number of speakers to try before giving up.

    Returns
    -------
    pd.Series or None
        The chosen target row, or None if no candidate was found.
    """
    if rejected_speakers is None:
        rejected_speakers = set()

    src_speaker = str(source_speaker_id)
    cands = target_df[target_df["mapped_age_class"] == target_age]

    if len(cands) == 0:
        # Fallback: any speaker different from source
        cands = target_df[target_df["speaker_id"].astype(str) != src_speaker]
        if len(cands) == 0:
            return None

    speakers = cands["speaker_id"].astype(str).unique().tolist()

    for _ in range(max_retries):
        # Filter speakers we have not rejected yet and not source
        allowed = [s for s in speakers
                   if s != src_speaker and s not in rejected_speakers]
        if not allowed:
            return None

        chosen_speaker = allowed[int(rng.integers(0, len(allowed)))]

        spk_files = cands[
            (cands["speaker_id"].astype(str) == chosen_speaker)
            & (cands["raw_duration_sec"] >= min_duration)
        ]

        if len(spk_files) == 0:
            rejected_speakers.add(chosen_speaker)
            continue

        chosen = spk_files.iloc[int(rng.integers(0, len(spk_files)))]
        return chosen

    return None


# =========================================================
# Manifest row building
# =========================================================

def row_to_manifest(row, kind, final_seg_path, transcript_lookup, setting_name):
    """Build one manifest row from a plan row and a saved final file."""
    is_replay = (kind == "replay")
    src_age   = safe_str(row.get("mapped_age_class"))
    tgt_age   = np.nan if is_replay else safe_str(row.get("target_age_class"))

    return {
        "source_seg_id":        safe_val(row.get("segment_id")),
        "parent_file_id":       safe_val(row.get("parent_file_id")),
        "target_file_id":       np.nan if is_replay else safe_val(row.get("target_file_id")),
        "start_sec":            safe_val(row.get("start_sec")),
        "end_sec":              safe_val(row.get("end_sec")),
        "source_seg_path":      safe_val(row.get("seg_path")),
        "source_file_path":     safe_val(row.get("source_file_path")),
        "target_file_path":     np.nan if is_replay else safe_val(row.get("target_file_path")),
        "source_speaker_id":    safe_val(row.get("speaker_id")),
        "source_gender":        safe_val(row.get("gender")),
        "source_dataset":       safe_val(row.get("dataset_source")),
        "target_dataset":       np.nan if is_replay else safe_val(row.get("target_dataset")),
        "source_age_class":     src_age,
        "target_speaker_id":    np.nan if is_replay else safe_val(row.get("target_speaker_id")),
        "target_gender":        np.nan if is_replay else safe_val(row.get("target_gender")),
        "target_age_class":     tgt_age,
        "authenticity":         "spoof",
        "spoof_type":           kind,
        "spoof_engine":         row["spoof_engine"],
        "cross_age_spoof":      False if is_replay else bool(row.get("cross_age_spoof", False)),
        "age_direction":        np.nan if is_replay else derive_age_direction(src_age, tgt_age),
        "source_transcript_id": safe_val(row.get("parent_file_id")) if kind == "tts" else np.nan,
        "source_transcript": (
            safe_val(transcript_lookup.get(str(row.get("segment_id", ""))))
            if kind == "tts" else np.nan
        ),
        "split":                safe_val(row.get("split")),
        "source_pool":          safe_val(row.get("pool")),
        "target_pool":          np.nan if is_replay else safe_val(row.get("target_pool")),
        "final_seg_path":       final_seg_path,
        "setting":              setting_name,
    }


def make_filename(row, kind):
    seg_id = str(row["segment_id"])
    return f"{kind}__{row['spoof_engine']}__{seg_id}.wav"


# =========================================================
# Main runner: one setting (train or valid)
# =========================================================

def run_setting(
    split_name,
    setting_name,
    source_df,
    target_df,
    transcript_lookup,
    settings_engines,
    out_base,
    manifest_base,
    progress_base,
    run_engine_fn,
    cross_age_p=CROSS_AGE_DEFAULT,
    seed=42,
    target_engine_filter=None,
):
    """
    Run one full (split, setting) end-to-end with dynamic assignment,
    target retries, engine fallback, and resume support.

    Parameters
    ----------
    split_name : str
        "train" or "valid".
    setting_name : str
        For example "set1".
    source_df : pd.DataFrame
        Source segments manifest for this split.
    target_df : pd.DataFrame
        Spoof targets manifest for this split.
    transcript_lookup : dict
        Mapping segment_id -> transcript text.
    settings_engines : list of str
        Engines that belong to this setting.
    out_base : str
        Base output directory for WAV files.
    manifest_base : str
        Base output directory for manifest CSV files.
    progress_base : str
        Base directory for progress and failed CSVs.
    run_engine_fn : callable
        Function with signature
            run_engine_fn(engine_name, source_path, target_path, text)
        that returns (audio, used_text). Same contract as in the
        existing notebook.
    cross_age_p : float
        Probability of cross-age within each non-replay engine.
    seed : int
        RNG seed for deterministic assignment.
    target_engine_filter : str or None
        If given, only run this single engine inside the setting.
        Useful when running engine-by-engine in Colab.
    """
    rng = np.random.default_rng(seed)

    vc_engines     = [e for e in settings_engines if e.endswith("_vc")]
    tts_engines    = [e for e in settings_engines if e.endswith("_tts")]
    replay_engines = [e for e in settings_engines if e.startswith("replay_")]

    valid_seg_ids = set(transcript_lookup.keys())

    progress_csv = os.path.join(
        progress_base, f"progress_{split_name}_{setting_name}.csv"
    )
    failed_csv = os.path.join(
        progress_base, f"failed_{split_name}_{setting_name}.csv"
    )
    tracker = ProgressTracker(progress_csv, failed_csv)

    manifest_path = os.path.join(
        manifest_base, f"{split_name}_spoof_{setting_name}_clean.csv"
    )
    create_empty_manifest(manifest_path)

    # =========================
    # Build initial pools and assignments
    # =========================
    vc_pool, tts_pool, rep_pool = build_source_pools(
        source_df, valid_seg_ids, rng
    )

    vc_plan  = assign_engines_and_cross_age(vc_pool,  vc_engines,     cross_age_p, rng)
    tts_plan = assign_engines_and_cross_age(tts_pool, tts_engines,    cross_age_p, rng)
    rep_plan = assign_engines_and_cross_age(rep_pool, replay_engines, cross_age_p, rng)

    # Group by engine for easier per-engine running
    by_engine = {e: [] for e in settings_engines}
    for r in vc_plan:
        by_engine[r["spoof_engine"]].append(("vc", r))
    for r in tts_plan:
        by_engine[r["spoof_engine"]].append(("tts", r))
    for r in rep_plan:
        by_engine[r["spoof_engine"]].append(("replay", r))

    # =========================
    # Process each engine
    # =========================
    engines_to_run = (
        [target_engine_filter]
        if target_engine_filter is not None
        else settings_engines
    )

    overall_ok      = 0
    overall_skipped = 0
    overall_failed  = 0

    for engine in engines_to_run:
        if engine not in by_engine:
            print(f"[{split_name}/{setting_name}] engine not in setting: {engine}")
            continue

        kind_rows = by_engine[engine]
        if not kind_rows:
            print(f"[{split_name}/{setting_name}] no rows for {engine}")
            continue

        kind = kind_rows[0][0]
        type_engines = (
            vc_engines if kind == "vc"
            else tts_engines if kind == "tts"
            else replay_engines
        )

        out_dir = os.path.join(out_base, split_name, setting_name, engine)
        os.makedirs(out_dir, exist_ok=True)

        print(
            f"\n========== {split_name}/{setting_name} | {engine} | "
            f"{len(kind_rows)} planned rows =========="
        )

        # Stack of segments queued for this engine (for fallback)
        queue = [r for _, r in kind_rows]

        ok_count      = 0
        skipped_count = 0
        fail_count    = 0

        with ManifestWriter(manifest_path, flush_every=5) as mw:
            pbar = tqdm(
                total=len(queue),
                desc=f"{split_name}/{setting_name}/{engine}",
            )

            while queue:
                row = queue.pop(0)
                seg_id = str(row["segment_id"])

                # Skip if this exact (segment, engine) already done
                if tracker.is_done(seg_id, engine):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Skip if this segment was already completed by ANY engine
                # of the same type (fallback resolved it elsewhere)
                if tracker.segment_completed_for_type(seg_id, type_engines):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Skip if this engine already failed on this segment
                if tracker.engine_already_failed(seg_id, engine):
                    pbar.update(1)
                    continue

                # =========================
                # Pick target (vc / tts)
                # =========================
                target_row = None
                rejected_speakers = set()

                if kind != "replay":
                    target_row = pick_target(
                        target_df,
                        target_age=row["target_age_class"],
                        source_speaker_id=row["speaker_id"],
                        rng=rng,
                        rejected_speakers=rejected_speakers,
                    )
                    if target_row is None:
                        msg = "no eligible target speaker found"
                        tracker.mark_engine_failed(seg_id, engine, msg, kind)
                        _reassign_segment(
                            row, engine, type_engines, by_engine, queue
                        )
                        fail_count += 1
                        pbar.update(1)
                        continue

                    row["target_file_path"]   = target_row["processed_path"]
                    row["target_speaker_id"]  = target_row["speaker_id"]
                    row["target_gender"]      = target_row.get("gender", np.nan)
                    row["target_pool"]        = target_row.get("pool", np.nan)
                    row["target_file_id"]     = target_row.get("file_id", np.nan)
                    row["target_dataset"]     = target_row.get("dataset", np.nan)

                # =========================
                # Try the engine, retrying with different speakers
                # =========================
                src_path = str(row["seg_path"])
                if not os.path.exists(src_path):
                    msg = f"source missing: {src_path}"
                    tracker.mark_engine_failed(seg_id, engine, msg, kind)
                    fail_count += 1
                    pbar.update(1)
                    continue

                text = (
                    transcript_lookup.get(seg_id)
                    if kind == "tts" else None
                )

                tgt_path = (
                    None if kind == "replay"
                    else str(row.get("target_file_path", ""))
                )

                out_path = os.path.join(out_dir, make_filename(row, kind))

                success = False
                last_err = None

                # Up to MAX_TARGET_RETRIES inner tries for vc / tts;
                # replay has no target to swap, so only 1 try.
                inner_tries = 1 if kind == "replay" else MAX_TARGET_RETRIES

                for attempt in range(inner_tries):
                    try:
                        audio, _ = run_engine_fn(
                            engine, src_path, tgt_path, text
                        )

                        save_padded(
                            out_path,
                            audio,
                            apply_vad=(kind == "tts"),
                        )

                        mw.append(row_to_manifest(
                            row=row,
                            kind=kind,
                            final_seg_path=out_path,
                            transcript_lookup=transcript_lookup,
                            setting_name=setting_name,
                        ))

                        tracker.mark_done(seg_id, engine, out_path)
                        ok_count += 1
                        success = True
                        break

                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"

                        # For vc/tts, swap to a different target speaker and retry
                        if kind != "replay" and target_row is not None:
                            rejected_speakers.add(str(target_row["speaker_id"]))
                            target_row = pick_target(
                                target_df,
                                target_age=row["target_age_class"],
                                source_speaker_id=row["speaker_id"],
                                rng=rng,
                                rejected_speakers=rejected_speakers,
                            )
                            if target_row is None:
                                break

                            row["target_file_path"]  = target_row["processed_path"]
                            row["target_speaker_id"] = target_row["speaker_id"]
                            row["target_gender"]     = target_row.get("gender", np.nan)
                            row["target_pool"]       = target_row.get("pool", np.nan)
                            row["target_file_id"]    = target_row.get("file_id", np.nan)
                            row["target_dataset"]    = target_row.get("dataset", np.nan)
                            tgt_path = str(row["target_file_path"])
                        else:
                            break

                if not success:
                    tracker.mark_engine_failed(
                        seg_id, engine, last_err or "unknown error", kind
                    )
                    _reassign_segment(
                        row, engine, type_engines, by_engine, queue
                    )
                    fail_count += 1
                    print(f"FAIL [{engine}] {seg_id} -> {last_err}")

                pbar.update(1)

            pbar.close()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[{split_name}/{setting_name}/{engine}] "
            f"ok={ok_count}, skipped={skipped_count}, failed={fail_count}"
        )

        overall_ok      += ok_count
        overall_skipped += skipped_count
        overall_failed  += fail_count

    return {
        "ok":      overall_ok,
        "skipped": overall_skipped,
        "failed":  overall_failed,
    }


def _reassign_segment(row, failing_engine, type_engines, by_engine, queue):
    """
    Move a failed segment to another engine of the same type so that
    the next time that engine is run, it picks up this segment.

    The segment is added to that engine's bucket in by_engine.
    The active queue is not modified here; the new engine will
    process the segment when its own run starts (or in the current
    run, if scheduled later).
    """
    alternatives = [e for e in type_engines if e != failing_engine]
    if not alternatives:
        return

    # Pick the engine with the fewest queued items, for balance
    chosen = min(alternatives, key=lambda e: len(by_engine.get(e, [])))
    kind_label = (
        "vc" if chosen.endswith("_vc")
        else "tts" if chosen.endswith("_tts")
        else "replay"
    )

    new_row = dict(row)
    new_row["spoof_engine"] = chosen

    by_engine.setdefault(chosen, []).append((kind_label, new_row))
