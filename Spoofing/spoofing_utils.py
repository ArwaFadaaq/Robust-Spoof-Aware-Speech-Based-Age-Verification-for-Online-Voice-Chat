import pandas as pd
import os
import glob

def read_trn_file(audio_path):
    """
    Given path to .wav, read corresponding .trn file
    """
    trn_path = os.path.splitext(audio_path)[0] + ".trn"
    if os.path.exists(trn_path):
        try:
            with open(trn_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except:
            return None
    return None


def build_transcript_inventory(
    source_csv,
    manifest_csv,
    metadata_csv,
    myst_root_dir,
    output_csv,
    age_col="mapped_age_class"
):

    # ===== Load files =====
    df_source = pd.read_csv(source_csv)
    df_manifest = pd.read_csv(manifest_csv, low_memory=False)
    df_meta = pd.read_csv(metadata_csv)

    # ===== Split datasets =====
    df_cv   = df_source[df_source["dataset_source"] == "common_voice"].copy()
    df_myst = df_source[df_source["dataset_source"] == "myst"].copy()

    print(f"CV rows: {len(df_cv)}")
    print(f"MYST rows: {len(df_myst)}")

    # =====================================================
    # =============== COMMON VOICE =========================
    # =====================================================

    df_cv_merged = df_cv.merge(
        df_manifest,
        left_on="parent_file_id",
        right_on="file_id",
        how="left"
    )

    df_cv_merged = df_cv_merged.merge(
        df_meta,
        left_on="path",
        right_on="clips_path",
        how="left"
    )

    result_cv = df_cv_merged[
        [
            "segment_id",
            "clips_path",
            "sentence",
            age_col,
            "dataset_source"
        ]
    ].copy()

    # =====================================================
    # =================== MYST =============================
    # =====================================================

    df_myst_merged = df_myst.merge(
        df_manifest,
        left_on="parent_file_id",
        right_on="file_id",
        how="left"
    )

    # ===== read transcripts from .trn =====
    sentences = []
    for path in df_myst_merged["path"]:
        if pd.isna(path):
            sentences.append(None)
        else:
            sentences.append(read_trn_file(path))

    df_myst_merged["sentence"] = sentences

    result_myst = df_myst_merged[
        [
            "segment_id",
            "path",
            "sentence",
            age_col,
            "dataset_source"
        ]
    ].copy()

    result_myst = result_myst.rename(columns={"path": "clips_path"})

    # =====================================================
    # ================= MERGE ALL ==========================
    # =====================================================

    result = pd.concat([result_cv, result_myst], ignore_index=True)

    # ===== Transcript flag =====
    result["has_transcript"] = result["sentence"].notna().astype(int)

    # ===== Save =====
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result.to_csv(output_csv, index=False)

    print("\n✅ Saved transcript file:")
    print(output_csv)
    print(f"Total rows: {len(result)}")
    print(f"With transcript: {result['has_transcript'].sum()}")
    print(f"Without transcript: {(result['has_transcript'] == 0).sum()}")

    return result
