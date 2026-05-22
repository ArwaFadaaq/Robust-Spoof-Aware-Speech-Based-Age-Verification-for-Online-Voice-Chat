import os, sys, gc, math, json, hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .spoofing_utils import (
    SEED, SR, TARGET_SEC, TARGET_DUR, MAX_TGT_TRIES, MIN_OUTPUT_SEC, MAX_TGT_RETRY,
    MANIFEST_COLUMNS,
    set_seed, engine_type,
    safe_val, safe_str,
    save_audio, extract_longest_voiced,
    run_engine, find_valid_target_file, pick_target,
    ManifestWriter, create_empty_manifest,
    make_filename, build_manifest_row,
    load_processed_set, save_processed_set,
    load_json_config, save_json_config,
)

# Cross-age probability specific to train/valid splits (80%)
CROSS_AGE_P = 0.8

# Each set defines 6 engines: 2 vc + 2 tts + 2 replay
SETTINGS = {
    'set1': ['openvoice_vc', 'koko_vc',      'koko_tts',       'xttsv2_tts',     'replay_c1', 'replay_c2'],
    'set2': ['seed_vc',      'koko_vc',      'chatterbox_tts', 'xttsv2_tts',     'replay_c3', 'replay_c2'],
    'set3': ['seed_vc',      'openvoice_vc', 'koko_tts',       'chatterbox_tts', 'replay_c1', 'replay_c3'],
}

# Module-level globals — populated by init()
SPLIT_PATHS    = None  # Dict of CSV paths keyed by split name
OUT_BASE       = None  # Root directory for all generated audio files
MANIFEST_BASE  = None  # Directory for manifest CSV files
JSON_BASE      = None  # Directory for JSON engine config files
PROCESSED_BASE = None  # Directory for processed-segment tracking files
SPOOFING_PATH  = None  # Path to the Spoofing package directory
SPOOFING_CORE  = None  # Path to the Spoofing/core directory
SPLIT_DATA     = None  # Dict holding loaded DataFrames and transcript lookups


# =================================================================
# PATH HELPERS — train/valid specific naming conventions
# =================================================================

# Returns the manifest CSV path for a given split and setting.
def get_manifest_path(split, setting):
    return os.path.join(MANIFEST_BASE, f'{split}_spoof_{setting}_clean.csv')

# Returns the audio output directory for a given split, setting, and engine.
def get_out_dir(split, setting, engine):
    return os.path.join(OUT_BASE, split, setting, engine)

# Returns the processed-segments JSON path for a given split and setting.
def get_processed_path(split, setting):
    return os.path.join(PROCESSED_BASE, f'{split}_{setting}_processed.json')

# Returns the JSON config file path for a given split and setting.
def get_json_config_path(split, setting):
    return os.path.join(JSON_BASE, f'{split}_{setting}_config.json')


# =================================================================
# INIT
# =================================================================

# Initialises all module-level globals, loads train/valid CSVs from Drive,
# pre-creates all output directories, and builds JSON configs if missing.
def init(project_root, repo_dir,
         train_source_csv, train_target_csv, train_transcript_csv,
         valid_source_csv, valid_target_csv, valid_transcript_csv,
         out_base, manifest_base, json_base, processed_base):

    global SPLIT_PATHS, OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE
    global SPOOFING_PATH, SPOOFING_CORE, SPLIT_DATA

    SPLIT_PATHS = {
        'train': {
            'source'    : train_source_csv,
            'target'    : train_target_csv,
            'transcript': train_transcript_csv,
        },
        'valid': {
            'source'    : valid_source_csv,
            'target'    : valid_target_csv,
            'transcript': valid_transcript_csv,
        },
    }

    OUT_BASE       = out_base
    MANIFEST_BASE  = manifest_base
    JSON_BASE      = json_base
    PROCESSED_BASE = processed_base

    # Create all base directories
    for d in [OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE]:
        os.makedirs(d, exist_ok=True)

    # Pre-create one output folder per split/setting/engine combination
    for split in ['train', 'valid']:
        for setting_name, engines in SETTINGS.items():
            for eng in engines:
                os.makedirs(os.path.join(OUT_BASE, split, setting_name, eng), exist_ok=True)

    SPOOFING_PATH = f'{repo_dir}/Spoofing'
    SPOOFING_CORE = f'{repo_dir}/Spoofing/core'

    set_seed(SEED)

    print(f'OUT_BASE:       {OUT_BASE}')
    print(f'MANIFEST_BASE:  {MANIFEST_BASE}')
    print(f'JSON_BASE:      {JSON_BASE}')
    print(f'PROCESSED_BASE: {PROCESSED_BASE}')
    print(f'CROSS_AGE_P={CROSS_AGE_P}, TARGET_DUR>={TARGET_DUR}s, TARGET_SEC={TARGET_SEC}s, MIN_OUTPUT_SEC={MIN_OUTPUT_SEC}s')

    train_src, train_tgt, train_tr_lookup = _load_split_data('train')
    valid_src, valid_tgt, valid_tr_lookup = _load_split_data('valid')

    print(f'TRAIN: {len(train_src)} source segs, {len(train_tgt)} target files, {len(train_tr_lookup)} transcripts')
    print(f'VALID: {len(valid_src)} source segs, {len(valid_tgt)} target files, {len(valid_tr_lookup)} transcripts')
    print(f'TRAIN age distribution:\n{train_src["mapped_age_class"].value_counts().to_string()}')
    print(f'VALID age distribution:\n{valid_src["mapped_age_class"].value_counts().to_string()}')

    SPLIT_DATA = {
        'train': {'src': train_src, 'tgt': train_tgt, 'tr': train_tr_lookup},
        'valid': {'src': valid_src, 'tgt': valid_tgt, 'tr': valid_tr_lookup},
    }

    # Build JSON configs for all splits and settings if not already present
    print('\nBuilding JSON configs...')
    for split in ['train', 'valid']:
        n_src = len(SPLIT_DATA[split]['src'])
        print(f'\n  [{split}] n_src={n_src}')
        for setting_name, engines in SETTINGS.items():
            _build_json_config(split, setting_name, engines, n_src, CROSS_AGE_P)

    print('\nAll JSON configs ready')


# =================================================================
# DATA LOADING
# =================================================================

# Reads source, target, and transcript CSVs for a given split from Drive
# and returns the DataFrames plus a segment_id -> transcript lookup dict.
def _load_split_data(split):
    src = pd.read_csv(SPLIT_PATHS[split]['source'],     dtype={'speaker_id': str})
    tgt = pd.read_csv(SPLIT_PATHS[split]['target'],     dtype={'speaker_id': str})
    tr  = pd.read_csv(SPLIT_PATHS[split]['transcript'], dtype={'segment_id': str})
    lookup = dict(zip(
        tr[tr['has_transcript'] == 1]['segment_id'].astype(str),
        tr[tr['has_transcript'] == 1]['sentence_clean']
    ))
    return src, tgt, lookup


# =================================================================
# JSON CONFIG BUILDER
# =================================================================

# Builds and saves the per-engine JSON config for one (split, setting).
# Divides the source pool evenly across vc / tts / replay engine types,
# then distributes within each type. Skips if the file already exists.
def _build_json_config(split, setting_name, engines, n_src, cross_age_p):
    json_path = get_json_config_path(split, setting_name)

    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f'  Config already exists for {split}/{setting_name} — skipping')
        for eng, v in existing.items():
            ca  = f", cross_age={v['cross_age_count']}"        if 'cross_age_count' in v else ''
            cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
            print(f'    {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')
        return existing, json_path

    vc_engines     = [e for e in engines if engine_type(e) == 'vc']
    tts_engines    = [e for e in engines if engine_type(e) == 'tts']
    replay_engines = [e for e in engines if engine_type(e) == 'replay']

    # Divide n_src into three equal parts; distribute remainder one-by-one
    n_per_type = n_src // 3
    remainder  = n_src - 3 * n_per_type
    n_vc     = n_per_type + (1 if remainder > 0 else 0)
    n_tts    = n_per_type + (1 if remainder > 1 else 0)
    n_replay = n_src - n_vc - n_tts

    # Distributes a total count as evenly as possible across n engines.
    def split_evenly(total, n):
        base = total // n
        rem  = total % n
        return [base + (1 if i < rem else 0) for i in range(n)]

    vc_counts     = split_evenly(n_vc,     len(vc_engines))     if vc_engines     else []
    tts_counts    = split_evenly(n_tts,    len(tts_engines))    if tts_engines    else []
    replay_counts = split_evenly(n_replay, len(replay_engines)) if replay_engines else []

    config = {}

    for eng, cnt in zip(vc_engines, vc_counts):
        config[eng] = {
            'count_required' : cnt,
            'cross_age_count': round(cnt * cross_age_p),
            'cross_age_done' : 0,
            'count_done'     : 0,
        }

    for eng, cnt in zip(tts_engines, tts_counts):
        config[eng] = {
            'count_required' : cnt,
            'cross_age_count': round(cnt * cross_age_p),
            'cross_age_done' : 0,
            'count_done'     : 0,
        }

    # Replay engines have no cross_age tracking
    for eng, cnt in zip(replay_engines, replay_counts):
        config[eng] = {
            'count_required': cnt,
            'count_done'    : 0,
        }

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'  New config created for {split}/{setting_name}')
    for eng, v in config.items():
        ca  = f", cross_age={v['cross_age_count']}"        if 'cross_age_count' in v else ''
        cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
        print(f'    {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')

    return config, json_path


# =================================================================
# STATUS CHECK
# =================================================================

# Prints a formatted progress bar summary for all splits and settings.
def status_check():
    print('\n' + '='*70)
    print('STATUS CHECK')
    print('='*70)
    for split in ['train', 'valid']:
        for setting_name in ['set1', 'set2', 'set3']:
            try:
                cfg = load_json_config(JSON_BASE, split, setting_name)
                print(f'\n[{split}/{setting_name}]')
                total_req = total_done = 0
                for eng, v in cfg.items():
                    req    = v['count_required']
                    done   = v['count_done']
                    pct    = 100 * done / req if req > 0 else 0
                    ca     = f" cross_age={v['cross_age_count']}" if 'cross_age_count' in v else ''
                    bar    = '#' * int(pct // 10) + '-' * (10 - int(pct // 10))
                    status = 'DONE' if done >= req else 'IN PROGRESS'
                    print(f'  [{status}] {eng:<18} [{bar}] {done:>5}/{req:<5} ({pct:5.1f}%){ca}')
                    total_req  += req
                    total_done += done
                pct_total = 100 * total_done / total_req if total_req > 0 else 0
                print(f'  TOTAL: {total_done}/{total_req} ({pct_total:.1f}%)')
            except FileNotFoundError:
                print(f'  Config not found for {split}/{setting_name}')


# =================================================================
# MAIN ENGINE RUNNER
# =================================================================

# Runs one engine for a given (split, setting) pair until count_required is reached.
#
# Flow:
#   1. Load JSON config and verify the engine is present and not already complete.
#   2. Shuffle the source pool deterministically; exclude already-processed segments.
#      For TTS, further filter to segments that have a valid transcript.
#   3. Derive a deterministic per-engine RNG seed from (split, setting, engine).
#   4. Loop over available sources:
#        a. Determine cross_age flag based on how many cross-age samples still needed.
#        b. Replay: run engine directly with no target.
#        c. VC/TTS: retry up to MAX_TGT_RETRY different target speakers for the same
#           segment if VAD finds no speech or output is shorter than MIN_OUTPUT_SEC.
#        d. Pad/trim and write audio to disk.
#        e. Append a row to the manifest and update progress counters.
#        f. Every 10 successes: flush JSON config and processed-set to disk.
#   5. Final flush and summary print.
def run_one_engine(split_name, setting_name, engine):
    kind      = engine_type(engine)
    src_df    = SPLIT_DATA[split_name]['src'].copy()
    tgt_df    = SPLIT_DATA[split_name]['tgt'].copy()
    tr_lookup = SPLIT_DATA[split_name]['tr']

    config = load_json_config(JSON_BASE, split_name, setting_name)
    if engine not in config:
        print(f'  {engine} not in JSON config for {split_name}/{setting_name}')
        return

    eng_cfg         = config[engine]
    count_required  = eng_cfg['count_required']
    count_done      = eng_cfg['count_done']
    cross_age_count = eng_cfg.get('cross_age_count', 0)
    cross_age_done  = eng_cfg.get('cross_age_done', 0)

    if count_done >= count_required:
        print(f'{split_name}/{setting_name}/{engine}: already complete ({count_done}/{count_required})')
        return

    remaining     = count_required - count_done
    processed_set = load_processed_set(PROCESSED_BASE, split_name, setting_name)
    manifest_path = get_manifest_path(split_name, setting_name)
    out_dir       = get_out_dir(split_name, setting_name, engine)
    os.makedirs(out_dir, exist_ok=True)
    create_empty_manifest(manifest_path)

    print(f'\n========== {split_name}/{setting_name} | {engine} | need={remaining}/{count_required} ==========')
    if kind != 'replay':
        print(f'  cross_age: {cross_age_done}/{cross_age_count} done so far')

    # Shuffle source pool deterministically and remove already-processed segments
    all_src_shuffled = src_df.sample(frac=1, random_state=SEED)
    available_src    = all_src_shuffled[
        ~all_src_shuffled['segment_id'].astype(str).isin(processed_set)
    ].reset_index(drop=True)

    # TTS needs a transcript — filter out segments without one
    if kind == 'tts':
        valid_ids     = set(tr_lookup.keys())
        available_src = available_src[
            available_src['segment_id'].astype(str).isin(valid_ids)
        ].reset_index(drop=True)

    # Derive a stable per-engine RNG seed for reproducible target selection
    seed_str    = f'{split_name}_{setting_name}_{engine}'
    engine_seed = SEED + int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng_target  = np.random.default_rng(engine_seed)

    src_tried     = 0
    src_idx       = 0
    ok_count      = 0
    ok_cross      = 0
    batch_counter = 0

    pbar = tqdm(total=remaining, desc=f'{split_name}/{setting_name}/{engine}')

    with ManifestWriter(manifest_path, flush_every=10) as mw:
        while ok_count < remaining and src_idx < len(available_src):

            src_row   = available_src.iloc[src_idx]
            src_idx  += 1
            src_tried += 1

            seg_id   = str(src_row.get('segment_id', ''))
            src_path = str(src_row.get('seg_path', ''))
            src_age  = safe_str(src_row.get('mapped_age_class'))

            if not src_path or not os.path.exists(src_path):
                continue

            # Replay has no target; for vc/tts decide cross_age based on quota remaining
            if kind == 'replay':
                cross_age = False
            else:
                cross_age = (cross_age_done + ok_cross) < cross_age_count

            tgt_row     = None
            final_audio = None

            # ---- Replay: no target needed, run directly ----
            if kind == 'replay':
                try:
                    raw_audio = run_engine(engine, src_path, None, None,
                                           spoofing_path=SPOOFING_PATH,
                                           spoofing_core=SPOOFING_CORE)
                except Exception as e:
                    print(f'  FAIL [{engine}] src={seg_id} -> {type(e).__name__}: {e}')
                    continue

                if raw_audio is None:
                    continue

                if hasattr(raw_audio, 'detach'):
                    final_audio = raw_audio.detach().cpu().numpy()
                else:
                    final_audio = np.asarray(raw_audio).squeeze().astype(np.float32)

            # ---- VC / TTS: retry up to MAX_TGT_RETRY target speakers ----
            else:
                text = None
                if kind == 'tts':
                    text = tr_lookup.get(seg_id)
                    if not text or (isinstance(text, float) and math.isnan(text)):
                        continue

                # Build candidate target pool respecting cross_age
                opposite_age = 'adult' if src_age == 'minor' else 'minor'
                desired_age  = opposite_age if cross_age else src_age
                age_pool     = tgt_df[
                    tgt_df['mapped_age_class'] == desired_age
                ]['speaker_id'].unique().copy()
                rng_target.shuffle(age_pool)

                saved = False

                for spk in age_pool[:MAX_TGT_RETRY]:
                    tgt_file = find_valid_target_file(tgt_df, spk, rng_target)
                    if not tgt_file or not os.path.exists(tgt_file):
                        continue

                    tgt_row_candidate = tgt_df[tgt_df['speaker_id'] == spk].iloc[0]

                    try:
                        raw_audio = run_engine(engine, src_path, tgt_file, text,
                                               spoofing_path=SPOOFING_PATH,
                                               spoofing_core=SPOOFING_CORE)
                    except Exception as e:
                        print(f'  FAIL [{engine}] src={seg_id} tgt={spk} -> {type(e).__name__}: {e}')
                        continue

                    if raw_audio is None:
                        continue

                    if hasattr(raw_audio, 'detach'):
                        raw_np = raw_audio.detach().cpu().numpy()
                    else:
                        raw_np = np.asarray(raw_audio).squeeze().astype(np.float32)

                    # TTS: extract longest voiced segment and check minimum duration
                    if kind == 'tts':
                        voiced, found = extract_longest_voiced(raw_np, sr=SR)
                        if not found:
                            print(f'  RETRY [{engine}] {seg_id}: VAD no speech with tgt={spk}')
                            continue
                        if len(voiced) / SR < MIN_OUTPUT_SEC:
                            print(f'  RETRY [{engine}] {seg_id}: voiced too short ({len(voiced)/SR:.2f}s) with tgt={spk}')
                            continue
                        candidate_audio = voiced

                    # VC: check minimum duration directly
                    else:
                        if len(raw_np) / SR < MIN_OUTPUT_SEC:
                            print(f'  RETRY [{engine}] {seg_id}: vc too short ({len(raw_np)/SR:.2f}s) with tgt={spk}')
                            continue
                        candidate_audio = raw_np

                    # Passed all checks — accept this target
                    final_audio = candidate_audio
                    tgt_row     = tgt_row_candidate
                    saved       = True
                    break

                if not saved:
                    print(f'  SKIP [{engine}] {seg_id}: all {MAX_TGT_RETRY} target candidates exhausted')
                    continue

            # ---- Save audio ----
            out_filename = make_filename(seg_id, engine, kind)
            out_path     = os.path.join(out_dir, out_filename)

            try:
                save_audio(out_path, final_audio, sr=SR, target_sec=TARGET_SEC)
            except Exception as e:
                print(f'  SAVE FAIL [{engine}] {seg_id} -> {e}')
                continue

            ok_count      += 1
            if cross_age:
                ok_cross  += 1
            processed_set.add(seg_id)
            batch_counter += 1
            print(f'[{engine}] saved: {seg_id} | cross_age={cross_age}')

            config[engine]['count_done'] = count_done + ok_count
            if kind != 'replay':
                config[engine]['cross_age_done'] = cross_age_done + ok_cross

            # Flush state to disk every 10 successful saves for resume safety
            if batch_counter % 10 == 0:
                save_json_config(JSON_BASE, split_name, setting_name, config)
                save_processed_set(PROCESSED_BASE, split_name, setting_name, processed_set)

            mw.append(build_manifest_row(
                src_row=src_row, tgt_row=tgt_row, kind=kind, eng=engine,
                cross_age=cross_age, spoofed_seg_path=out_path,
                tr_lookup=tr_lookup, split=split_name, setting=setting_name
            ))

            pbar.update(1)

    pbar.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final save regardless of batch_counter
    save_json_config(JSON_BASE, split_name, setting_name, config)
    save_processed_set(PROCESSED_BASE, split_name, setting_name, processed_set)

    final_done       = count_done + ok_count
    final_cross_done = cross_age_done + ok_cross
    print(f'{split_name}/{setting_name}/{engine}: done={ok_count}, total={final_done}/{count_required}, '
          f'cross_age={final_cross_done}/{cross_age_count}, src_tried={src_tried}')

    if final_done < count_required:
        print('  Source pool exhausted before reaching target count.')
    if kind != 'replay' and final_cross_done != cross_age_count:
        print(f'  cross_age mismatch: got {final_cross_done}, expected {cross_age_count}')
