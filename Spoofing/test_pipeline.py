"""
Test Spoofing Pipeline (Dynamic, Resumable)

Differences from train / valid
-------------------------------
1. For every source segment in test_spoof_source, three spoofed
   outputs are generated: one VC, one TTS, one Replay.
   Therefore the three test manifests each have the same size as
   the source set.

2. Three engines per type (instead of two).
       VC     : seed_vc, koko_vc, openvoice_vc
       TTS    : koko_tts, xtts_v2, chatterbox_tts
       Replay : replay_c1, replay_c2, replay_c3

3. Within each type, the three engines receive equal shares of the
   source segments.

4. Cross-age probability is 0.5 for VC and TTS (instead of 0.8).
   Replay is always cross_age = False.

5. Each spoof type produces its own manifest:
       test_spoof_vc_clean.csv
       test_spoof_tts_clean.csv
       test_spoof_replay_clean.csv

6. Progress and failure tracking is per spoof type.

7. Engine fallback within the same type is supported (same logic as
   train_valid_pipeline.run_setting).
"""

import os
import math
import gc
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
    assign_engines_balanced,
    TARGET_SR,
)

from .train_valid_pipeline import (
    ProgressTracker,
    pick_target,
    row_to_manifest,
    make_filename,
    FLIP_AGE,
    MIN_TARGET_DURATION,
    MAX_TARGET_RETRIES,
)


# =========================================================
# Defaults
# =========================================================

CROSS_AGE_P_TEST = 0.5
SEED_DEFAULT     = 123


# =========================================================
# Test type runner
# =========================================================

def run_test_type(
    spoof_type,
    engines,
    source_df,
    target_df,
    transcript_lookup,
    out_base,
    manifest_base,
    progress_base,
    run_engine_fn,
    cross_age_p=CROSS_AGE_P_TEST,
    seed=SEED_DEFAULT,
    target_engine_filter=None,
):
    """
    Run one spoof type for the test split.

    Parameters
    ----------
    spoof_type : str
        "vc", "tts", or "replay".
    engines : list of str
        Engines belonging to this spoof type (3 engines for test).
    source_df : pd.DataFrame
        Full test_spoof_source_clean dataframe. Every segment will be
        processed exactly once for this spoof type.
    target_df : pd.DataFrame
        Test spoof targets manifest.
    transcript_lookup : dict
        segment_id -> transcript text (only used for tts).
    out_base : str
        Base directory for WAV output. Files go under
        out_base/test/{spoof_type}/{engine}/.
    manifest_base : str
        Directory for manifest CSV files.
        Manifest file is named test_spoof_{spoof_type}_clean.csv.
    progress_base : str
        Directory for progress and failed CSV files.
    run_engine_fn : callable
        Same engine dispatcher used by the train / valid pipeline.
    cross_age_p : float
        Cross-age probability for VC and TTS (0.5 by default).
    seed : int
        RNG seed for assignment.
    target_engine_filter : str or None
        If set, only run this one engine; useful for engine-by-engine
        execution from a notebook.

    Returns
    -------
    dict
        Counters: ok / skipped / failed.
    """
    if spoof_type not in ("vc", "tts", "replay"):
        raise ValueError(f"Unknown spoof_type: {spoof_type}")

    is_replay = (spoof_type == "replay")
    is_tts    = (spoof_type == "tts")

    rng = np.random.default_rng(seed)

    valid_seg_ids = set(transcript_lookup.keys())

    # =========================
    # Build the pool
    # =========================
    if is_tts:
        # TTS needs a transcript per segment
        df = source_df[source_df["segment_id"].astype(str).isin(valid_seg_ids)].copy()
        if len(df) < len(source_df):
            print(f"[test/tts] {len(source_df) - len(df)} segments dropped "
                  f"due to missing transcript")
    else:
        df = source_df.copy()

    # Shuffle deterministically so engine assignment is reproducible
    df = df.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    pool = df.to_dict("records")

    # =========================
    # Assign engines and cross-age
    # =========================
    plan = assign_engines_balanced(
        pool=pool,
        engines=engines,
        cross_age_p=cross_age_p,
        rng=rng,
        flip_age_map=FLIP_AGE,
        is_replay=is_replay,
    )

    by_engine = {e: [] for e in engines}
    for r in plan:
        by_engine[r["spoof_engine"]].append(r)

    # =========================
    # Resume + manifest setup
    # =========================
    progress_csv = os.path.join(progress_base, f"progress_test_{spoof_type}.csv")
    failed_csv   = os.path.join(progress_base, f"failed_test_{spoof_type}.csv")
    tracker      = ProgressTracker(progress_csv, failed_csv)

    manifest_path = os.path.join(manifest_base, f"test_spoof_{spoof_type}_clean.csv")
    create_empty_manifest(manifest_path)

    # =========================
    # Run engines
    # =========================
    engines_to_run = (
        [target_engine_filter] if target_engine_filter is not None else engines
    )

    overall_ok      = 0
    overall_skipped = 0
    overall_failed  = 0

    for engine in engines_to_run:
        if engine not in by_engine:
            print(f"[test/{spoof_type}] engine not in this type: {engine}")
            continue

        queue = list(by_engine[engine])
        if not queue:
            print(f"[test/{spoof_type}] no rows for {engine}")
            continue

        out_dir = os.path.join(out_base, "test", spoof_type, engine)
        os.makedirs(out_dir, exist_ok=True)

        print(
            f"\n========== test/{spoof_type} | {engine} | "
            f"{len(queue)} planned rows =========="
        )

        ok_count      = 0
        skipped_count = 0
        fail_count    = 0

        with ManifestWriter(manifest_path, flush_every=5) as mw:
            pbar = tqdm(total=len(queue), desc=f"test/{spoof_type}/{engine}")

            while queue:
                row = queue.pop(0)
                seg_id = str(row["segment_id"])

                # Skip if this exact (segment, engine) already done
                if tracker.is_done(seg_id, engine):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Skip if any engine of this type already completed it
                if tracker.segment_completed_for_type(seg_id, engines):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                # Skip if this engine already failed on this segment
                if tracker.engine_already_failed(seg_id, engine):
                    pbar.update(1)
                    continue

                # =========================
                # Target selection (vc / tts only)
                # =========================
                target_row = None
                rejected_speakers = set()

                if not is_replay:
                    target_row = pick_target(
                        target_df,
                        target_age=row["target_age_class"],
                        source_speaker_id=row["speaker_id"],
                        rng=rng,
                        rejected_speakers=rejected_speakers,
                    )
                    if target_row is None:
                        tracker.mark_engine_failed(
                            seg_id, engine,
                            "no eligible target speaker", spoof_type,
                        )
                        _reassign(row, engine, engines, by_engine)
                        fail_count += 1
                        pbar.update(1)
                        continue

                    row["target_file_path"]  = target_row["processed_path"]
                    row["target_speaker_id"] = target_row["speaker_id"]
                    row["target_gender"]     = target_row.get("gender", np.nan)
                    row["target_pool"]       = target_row.get("pool", np.nan)
                    row["target_file_id"]    = target_row.get("file_id", np.nan)
                    row["target_dataset"]    = target_row.get("dataset", np.nan)

                # =========================
                # Engine call (with target retries for vc / tts)
                # =========================
                src_path = str(row["seg_path"])
                if not os.path.exists(src_path):
                    tracker.mark_engine_failed(
                        seg_id, engine,
                        f"source missing: {src_path}", spoof_type,
                    )
                    fail_count += 1
                    pbar.update(1)
                    continue

                text     = transcript_lookup.get(seg_id) if is_tts else None
                tgt_path = None if is_replay else str(row.get("target_file_path", ""))
                out_path = os.path.join(out_dir, make_filename(row, spoof_type))

                success     = False
                last_err    = None
                inner_tries = 1 if is_replay else MAX_TARGET_RETRIES

                for _ in range(inner_tries):
                    try:
                        audio, _ = run_engine_fn(engine, src_path, tgt_path, text)

                        save_padded(out_path, audio, apply_vad=is_tts)

                        mw.append(row_to_manifest(
                            row=row,
                            kind=spoof_type,
                            final_seg_path=out_path,
                            transcript_lookup=transcript_lookup,
                            setting_name="test",
                        ))

                        tracker.mark_done(seg_id, engine, out_path)
                        ok_count += 1
                        success = True
                        break

                    except Exception as e:
                        last_err = f"{type(e).__name__}: {e}"

                        if not is_replay and target_row is not None:
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
                        seg_id, engine, last_err or "unknown error", spoof_type,
                    )
                    _reassign(row, engine, engines, by_engine)
                    fail_count += 1
                    print(f"FAIL [test/{spoof_type}/{engine}] {seg_id} -> {last_err}")

                pbar.update(1)

            pbar.close()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[test/{spoof_type}/{engine}] "
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


def _reassign(row, failing_engine, type_engines, by_engine):
    """Move a failed segment to the engine of the same type with the smallest queue."""
    alternatives = [e for e in type_engines if e != failing_engine]
    if not alternatives:
        return
    chosen = min(alternatives, key=lambda e: len(by_engine.get(e, [])))
    new_row = dict(row)
    new_row["spoof_engine"] = chosen
    by_engine.setdefault(chosen, []).append(new_row)


# =========================================================
# Backward-compatible alias
# =========================================================

def run_test_setting(*args, **kwargs):
    """Alias kept for compatibility with the old placeholder name."""
    return run_test_type(*args, **kwargs)
