import pandas as pd

def build_transcript_inventory(
    source_csv,
    manifest_csv,
    metadata_csv,
    output_csv
):
    """
    Builds transcript inventory by linking:
    segment_id -> parent_file_id -> path -> clips_path -> sentence
    """

    # Load files
    df_source = pd.read_csv(source_csv)
    df_manifest = pd.read_csv(manifest_csv)
    df_meta = pd.read_csv(metadata_csv)

    # Step 1: parent_file_id -> path
    df_merged = df_source.merge(
        df_manifest,
        left_on="parent_file_id",
        right_on="file_id",
        how="left"
    )

    # Step 2: path -> clips_path + sentence
    df_merged = df_merged.merge(
        df_meta,
        left_on="path",
        right_on="clips_path",
        how="left"
    )

    # Step 3: select final columns (keep all, even missing)
    result = df_merged[
        [
            "segment_id",
            "sentence_id",
            "sentence",
            "clips_path"
        ]
    ]

    # Step 4: DO NOT drop NaN (important for spoofing pipeline)

    # Step 5: add safety/debug flag
    result["has_transcript"] = result["sentence"].notna()

    # Step 6: optional sanity check column (useful later for debugging)
    result["is_matched"] = result["clips_path"].notna()

    # Save
    result.to_csv(output_csv, index=False)

    print(f"Saved transcript inventory to: {output_csv}")
    print(f"Total rows: {len(result)}")
    print(f"Matched transcripts: {result['has_transcript'].sum()}")
    print(f"Unmatched transcripts: {(~result['has_transcript']).sum()}")

    return result
