import pandas as pd
import os
import re


def clean_transcript(text):
    """
    Remove unwanted symbols and noise tags from raw transcript text.
    Returns None if the result is empty or too short.
    """
    if not text or not isinstance(text, str):
        return None

    # 1) Remove tagged noise/events: <laugh>, <breath>, <unk>, etc.
    text = re.sub(r'<[^>]+>', '', text)

    # 2) Remove bracketed annotations: [noise], [laughter], etc.
    text = re.sub(r'\[[^\]]+\]', '', text)

    # 3) Remove parenthetical notes: (cough), (inaudible), etc.
    text = re.sub(r'\([^)]+\)', '', text)

    # 4) Remove angle bracket residues: >>, <<, >, 
    text = re.sub(r'>>|<<|>|<', '', text)

    # 5) Remove special symbols: * % # @ ^ ~ ` | \ / = + { } [ ]
    text = re.sub(r'[*%#@^~`|\\\/=+{}\[\]]', '', text)

    # 6) Remove standalone hyphens (not between word characters)
    text = re.sub(r'(?<!\w)-(?!\w)', '', text)

    # 7) Remove ellipses and repeated dots
    text = re.sub(r'\.{2,}', '', text)

    # 8) Keep only: letters, digits, spaces, and: , . ! ? ' -
    text = re.sub(r"[^a-zA-Z0-9\s,\.!\?'\-]", '', text)

    # 9) Collapse multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    # Discard result if empty or too short to be useful
    if not text or len(text) < 3:
        return None

    return text


def truncate_to_phoneme_limit(text, max_chars=170):
    """
    Truncate text to fit within the TTS phoneme limit (~510 phonemes ≈ 170 chars).
    Tries to cut at a natural punctuation boundary to preserve sentence integrity.
    """
    if not text:
        return None

    # Return as-is if already within the limit
    if len(text) <= max_chars:
        return text

    # Try to cut at the last punctuation mark before the limit
    cut_point = None
    for punct in ['. ', '? ', '! ', ', ']:
        idx = text.rfind(punct, 0, max_chars)
        if idx != -1:
            cut_point = idx + len(punct)
            break

    # Fall back to the last space before the limit
    if not cut_point:
        cut_point = text.rfind(' ', 0, max_chars)

    # Hard cut if no boundary found
    if not cut_point or cut_point <= 0:
        cut_point = max_chars

    return text[:cut_point].strip()


def preprocess_transcript(raw_text, max_chars=170):
    """
    Full preprocessing pipeline: clean then truncate.
    Returns None if the text is unusable after cleaning.
    """
    cleaned = clean_transcript(raw_text)
    if not cleaned:
        return None
    return truncate_to_phoneme_limit(cleaned, max_chars=max_chars)


def read_trn_file(audio_path):
    """
    Read the .trn transcript file corresponding to a given .wav file path.
    Returns the transcript text, or None if the file doesn't exist or can't be read.
    """
    trn_path = os.path.splitext(audio_path)[0] + ".trn"
    if os.path.exists(trn_path):
        try:
            with open(trn_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def build_transcript_inventory(
    source_csv,
    manifest_csv,
    metadata_csv,
    myst_root_dir,
    output_csv,
    age_col="mapped_age_class",
    max_phoneme_chars=170
):
    """
    Build a transcript inventory CSV by joining source segments with their
    transcripts from Common Voice (metadata CSV) and MyST (.trn files).

    Output columns:
        segment_id, clips_path,
        sentence_original,   <- raw transcript
        sentence_clean,      <- cleaned + truncated transcript
        <age_col>, dataset_source, has_transcript
    """

    # ------------------------------------------------------------------
    # Load input files
    # ------------------------------------------------------------------
    df_source   = pd.read_csv(source_csv)
    df_manifest = pd.read_csv(manifest_csv, low_memory=False)
    df_meta     = pd.read_csv(metadata_csv)

    # ------------------------------------------------------------------
    # Split by dataset source
    # ------------------------------------------------------------------
    df_cv   = df_source[df_source["dataset_source"] == "common_voice"].copy()
    df_myst = df_source[df_source["dataset_source"] == "myst"].copy()
    print(f"CV rows:   {len(df_cv)}")
    print(f"MYST rows: {len(df_myst)}")

    # ------------------------------------------------------------------
    # Common Voice — transcript comes from the metadata CSV
    # ------------------------------------------------------------------
    df_cv_merged = df_cv.merge(
        df_manifest, left_on="parent_file_id", right_on="file_id", how="left"
    )
    df_cv_merged = df_cv_merged.merge(
        df_meta, left_on="path", right_on="clips_path", how="left"
    )
    result_cv = df_cv_merged[[
        "segment_id", "clips_path", "sentence", age_col, "dataset_source"
    ]].copy()

    # ------------------------------------------------------------------
    # MyST — transcript comes from .trn files on disk
    # ------------------------------------------------------------------
    df_myst_merged = df_myst.merge(
        df_manifest, left_on="parent_file_id", right_on="file_id", how="left"
    )

    sentences = []
    for path in df_myst_merged["path"]:
        if pd.isna(path):
            sentences.append(None)
        else:
            sentences.append(read_trn_file(path))
    df_myst_merged["sentence"] = sentences

    result_myst = df_myst_merged[[
        "segment_id", "path", "sentence", age_col, "dataset_source"
    ]].copy()
    result_myst = result_myst.rename(columns={"path": "clips_path"})

    # ------------------------------------------------------------------
    # Combine both datasets
    # ------------------------------------------------------------------
    result = pd.concat([result_cv, result_myst], ignore_index=True)

    # ------------------------------------------------------------------
    # Preprocessing — clean and truncate transcripts
    # ------------------------------------------------------------------
    print("\n Processing transcripts...")

    # Keep the original transcript in a separate column
    result.rename(columns={"sentence": "sentence_original"}, inplace=True)

    # Apply full preprocessing pipeline to produce the clean column
    result["sentence_clean"] = result["sentence_original"].apply(
        lambda x: preprocess_transcript(x, max_chars=max_phoneme_chars)
    )

    # Flag rows that have a usable transcript after cleaning
    result["has_transcript"] = result["sentence_clean"].notna().astype(int)

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    cols = [
        "segment_id",
        "clips_path",
        "sentence_original",
        "sentence_clean",
        age_col,
        "dataset_source",
        "has_transcript",
    ]
    result[cols].to_csv(output_csv, index=False)

    print("\n Saved transcript inventory:")
    print(f"   Path:             {output_csv}")
    print(f"   Total rows:       {len(result)}")
    print(f"   With transcript:  {result['has_transcript'].sum()}")
    print(f"   Without:          {(result['has_transcript'] == 0).sum()}")

    had_original = result["sentence_original"].notna().sum()
    had_clean    = result["sentence_clean"].notna().sum()
    print(f"\n Cleaning stats:")
    print(f"   Had raw text:      {had_original}")
    print(f"   Survived cleaning: {had_clean}")
    print(f"   Lost to cleaning:  {had_original - had_clean}")

    return result[cols]
