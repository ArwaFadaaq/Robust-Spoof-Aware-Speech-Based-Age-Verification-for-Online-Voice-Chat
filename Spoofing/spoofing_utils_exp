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
# ENGINE BALANCE (33/33/33)
# =========================================================

def balance_engines(df, engines, rng):
    n = len(df)
    base = engines * (n // len(engines) + 1)
    base = base[:n]
    rng.shuffle(base)
    df["spoof_engine"] = base
    return df

# =========================================================
# CROSS AGE RULES
# =========================================================

def assign_cross_age(df, p, rng):
    df = df.copy()
    df["cross_age_spoof"] = rng.choice([True, False], len(df), p=[p, 1-p])
    return df


def assign_target_age(df):
    df = df.copy()
    mask = df["cross_age_spoof"] == True

    df.loc[mask, "target_age_class"] = df.loc[mask, "source_age_class"].map(AGE_FLIP)
    df.loc[~mask, "target_age_class"] = df.loc[~mask, "source_age_class"]

    return df

# =========================================================
# TARGET SAMPLING
# =========================================================

def sample_target(tgt_df, src_row, rng):
    pool = tgt_df[tgt_df["mapped_age_class"] == src_row["target_age_class"]]

    if len(pool) == 0:
        pool = tgt_df

    return pool.iloc[int(rng.integers(0, len(pool)))]

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

def build_train_val(src, tr, rng, setting_cfg):
    df = src.copy()

    # TTS transcript filter
    df = filter_tts_segments(df, tr)

    # 33/33/33 spoof type
    df["spoof_type"] = rng.choice(SPOOF_TYPES, len(df))

    # engines balanced
    for t in SPOOF_TYPES:
        mask = df["spoof_type"] == t
        if mask.sum() > 0:
            df.loc[mask, "spoof_engine"] = balance_engines(
                df.loc[mask].copy(),
                ENGINE_MAP[t],
                rng
            )["spoof_engine"].values

    # cross-age 80/20 for VC & TTS only
    mask_ct = df["spoof_type"].isin(["vc", "tts"])
    df.loc[mask_ct, "cross_age_spoof"] = rng.choice(
        [True, False],
        mask_ct.sum(),
        p=[setting_cfg["cross_p"], 1-setting_cfg["cross_p"]]
    )

    df.loc[~mask_ct, "cross_age_spoof"] = np.nan

    df = assign_target_age(df)

    return df

# =========================================================
# TEST BUILDER
# =========================================================

def build_test(src, kind, rng):
    df = src.copy()
    df["spoof_type"] = kind

    # 33/33/33 engines
    df = balance_engines(df, ENGINE_MAP[kind], rng)

    if kind in ["vc", "tts"]:
        df["cross_age_spoof"] = rng.choice([True, False], len(df), p=[0.5, 0.5])
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
        "source_seg_id": row["segment_id"],
        "parent_file_id": row["parent_file_id"],

        "start_sec": row["start_sec"],
        "end_sec": row["end_sec"],

        "source_seg_path": row["seg_path"],
        "source_file_path": row["source_file_path"],
        "target_file_path": None if is_replay else row.get("target_file_path"),

        "source_speaker_id": row["speaker_id"],
        "source_gender": row.get("gender"),
        "dataset_source": row.get("dataset_source"),
        "source_age_class": row["source_age_class"],

        "target_speaker_id": None if is_replay else row.get("target_speaker_id"),
        "target_gender": None if is_replay else row.get("target_gender"),
        "target_age_class": None if is_replay else row.get("target_age_class"),

        "authenticity": "spoof",
        "spoof_type": kind,
        "spoof_engine": row["spoof_engine"],

        "cross_age_spoof": None if is_replay else row.get("cross_age_spoof"),
        "age_direction": None if is_replay else age_direction(
            row["source_age_class"],
            row.get("target_age_class")
        ),

        "source_transcript_id": row["parent_file_id"] if is_tts else np.nan,
        "source_transcript": transcript_map.get(str(row["segment_id"])) if is_tts else np.nan,

        "split": row.get("split"),
        "source_pool": row.get("pool"),
        "target_pool": None if is_replay else row.get("target_pool"),

        "final_seg_path": final_path
    }


def save_manifest(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

# =========================================================
# FINAL VALIDATION
# =========================================================

def validate_pipeline(src, out):
    return {
        "total_src": len(src),
        "total_out": len(out),
        "coverage": len(src) == len(out)
    }

# =========================================================
# SETTINGS CONFIG
# =========================================================

def get_setting_config(setting_id):
    return {
        1: {"cross_p": 0.8},
        2: {"cross_p": 0.5},
        3: {"cross_p": 0.2},
    }[setting_id]
