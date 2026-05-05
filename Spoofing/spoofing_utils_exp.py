# -*- coding: utf-8 -*-
"""spoofing_utils_exp (CLEAN FINAL FIXED VERSION)"""

import os
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

SR = 16000

SPOOF_TYPES = ["vc", "tts", "replay"]

ENGINE_MAP = {
    "vc": ["seed_vc", "koko_vc", "openvoice_vc"],
    "tts": ["koko_tts", "f5_tts", "chatterbox_tts"],
    "replay": ["replay_c1", "replay_c2", "replay_c3"]
}

AGE_FLIP = {"adult": "minor", "minor": "adult"}

# =========================================================
# FIXED CROSS AGE SETTINGS
# =========================================================

TRAIN_VAL_CROSS_P = 0.8
TEST_CROSS_P = 0.5

# =========================================================
# PATHS
# =========================================================

def build_setting_paths(base, setting_id, split):
    return {
        "data_dir": os.path.join(base, f"setting_{setting_id}", split),
        "manifest_path": os.path.join(base, "manifests", f"setting_{setting_id}_{split}.csv")
    }


def build_test_paths(base):
    return {
        "vc": os.path.join(base, "test", "vc"),
        "tts": os.path.join(base, "test", "tts"),
        "replay": os.path.join(base, "test", "replay"),
        "manifest": os.path.join(base, "manifests", "test")
    }

# =========================================================
# TRANSCRIPTS
# =========================================================

def filter_tts_segments(src, tr):
    valid = set(tr[tr["has_transcript"] == 1]["segment_id"])
    return src[src["segment_id"].isin(valid)].copy()


def build_transcript_map(tr):
    return dict(zip(tr["segment_id"].astype(str), tr["sentence"]))

# =========================================================
# 🔥 FIX: TTS SOURCE FILTER (common_voice + myst + transcript check)
# =========================================================

def filter_tts_source_segments(src, tr):
    allowed_sources = ["common_voice", "myst"]

    src = src[src["dataset_source"].isin(allowed_sources)].copy()

    valid = set(tr[tr["has_transcript"] == 1]["segment_id"])

    src = src[src["segment_id"].isin(valid)].copy()

    return src

# =========================================================
# ENGINE BALANCE
# =========================================================

def balance_engines(df, engines, rng):
    n = len(df)
    base = engines * (n // len(engines) + 1)
    base = base[:n]
    rng.shuffle(base)
    df["spoof_engine"] = base
    return df

# =========================================================
# CROSS AGE
# =========================================================

def assign_target_age(df):
    df = df.copy()
    mask = df["cross_age_spoof"] == True

    df.loc[mask, "target_age_class"] = df.loc[mask, "source_age_class"].map(AGE_FLIP)
    df.loc[~mask, "target_age_class"] = df.loc[~mask, "source_age_class"]

    return df

# =========================================================
# AGE DIRECTION
# =========================================================

def age_direction(src, tgt):
    if pd.isna(src) or pd.isna(tgt):
        return np.nan
    return f"{src[0]}2{tgt[0]}"

# =========================================================
# COVERAGE CHECK
# =========================================================

def coverage_check(src, out):
    missing = set(src["segment_id"]) - set(out["source_seg_id"])
    return {
        "missing": len(missing),
        "sample_missing": list(missing)[:20]
    }

# =========================================================
# TRAIN / VAL BUILDER
# =========================================================

def build_train_val(src, tr, rng):
    df = src.copy()

    if "seg_path" in df.columns:
        df = df.rename(columns={"seg_path": "source_seg_path"})

    # 🔥 APPLY TTS FILTER ONLY HERE
    df = filter_tts_source_segments(df, tr)

    df["spoof_type"] = rng.choice(SPOOF_TYPES, len(df))

    for t in SPOOF_TYPES:
        mask = df["spoof_type"] == t
        if mask.sum() > 0:
            df.loc[mask, "spoof_engine"] = balance_engines(
                df.loc[mask].copy(),
                ENGINE_MAP[t],
                rng
            )["spoof_engine"].values

    mask_ct = df["spoof_type"].isin(["vc", "tts"])
    df.loc[mask_ct, "cross_age_spoof"] = rng.choice(
        [True, False],
        mask_ct.sum(),
        p=[TRAIN_VAL_CROSS_P, 1-TRAIN_VAL_CROSS_P]
    )

    df.loc[~mask_ct, "cross_age_spoof"] = np.nan

    df = assign_target_age(df)

    return df

# =========================================================
# TEST BUILDER
# =========================================================

def build_test(src, kind, rng):
    df = src.copy()

    if "seg_path" in df.columns:
        df = df.rename(columns={"seg_path": "source_seg_path"})

    df["spoof_type"] = kind

    df = balance_engines(df, ENGINE_MAP[kind], rng)

    if kind in ["vc", "tts"]:
        df["cross_age_spoof"] = rng.choice(
            [True, False],
            len(df),
            p=[TEST_CROSS_P, 1-TEST_CROSS_P]
        )
        df = assign_target_age(df)
    else:
        df["cross_age_spoof"] = np.nan
        df["target_age_class"] = np.nan

    return df

# =========================================================
# MANIFEST
# =========================================================

def make_filename(row):
    return f"{row['spoof_engine']}__{row['source_seg_id']}.wav"


def build_manifest_row(row, final_path, transcript_map, kind):
    is_tts = kind == "tts"
    is_replay = kind == "replay"

    return {
        "source_seg_id": row.get("source_seg_id", np.nan),

        "parent_file_id": row.get("parent_file_id", np.nan),

        "start_sec": row.get("start_sec", np.nan),
        "end_sec": row.get("end_sec", np.nan),

        "source_seg_path": row.get("source_seg_path", np.nan),
        "source_file_path": row.get("source_file_path", np.nan),

        "target_file_path": np.nan if is_replay else row.get("target_file_path", np.nan),

        "source_speaker_id": row.get("speaker_id", np.nan),
        "source_gender": row.get("gender", np.nan),
        "dataset_source": row.get("dataset_source", np.nan),
        "source_age_class": row.get("source_age_class", np.nan),

        "target_speaker_id": np.nan if is_replay else row.get("target_speaker_id", np.nan),
        "target_gender": np.nan if is_replay else row.get("target_gender", np.nan),
        "target_age_class": np.nan if is_replay else row.get("target_age_class", np.nan),

        "authenticity": "spoof",
        "spoof_type": kind,
        "spoof_engine": row.get("spoof_engine", np.nan),

        "cross_age_spoof": np.nan if is_replay else row.get("cross_age_spoof", np.nan),
        "age_direction": np.nan if is_replay else age_direction(
            row.get("source_age_class"),
            row.get("target_age_class")
        ),

        "source_transcript_id": row.get("parent_file_id") if is_tts else np.nan,
        "source_transcript": transcript_map.get(str(row.get("source_seg_id", ""))) if is_tts else np.nan,

        "split": row.get("split", np.nan),
        "source_pool": row.get("pool", np.nan),
        "target_pool": np.nan if is_replay else row.get("target_pool", np.nan),

        "final_seg_path": final_path
    }

# =========================================================
# SAVE
# =========================================================

def save_manifest(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# =========================================================
# VALIDATION
# =========================================================

def validate_pipeline(src, out):
    return {
        "total_src": len(src),
        "total_out": len(out),
        "coverage": len(src) == len(out)
    }

# =========================================================
# SETTINGS
# =========================================================

def get_setting_config(setting_id):
    return {"note": "cross-age handled globally (train=0.8, test=0.5)"}
