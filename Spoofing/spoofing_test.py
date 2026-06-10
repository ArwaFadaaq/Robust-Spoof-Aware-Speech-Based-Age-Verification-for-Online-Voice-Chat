import os, sys, gc, math, json, hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .spoofing_utils import (
    SEED, SR, TARGET_SEC, TARGET_DUR, MIN_OUTPUT_SEC, MAX_TGT_RETRY,
    MANIFEST_COLUMNS,
    set_seed, engine_type,
    safe_val, safe_str,
    save_audio, extract_longest_voiced,
    run_engine, find_valid_target_file,
    ManifestWriter, create_empty_manifest,
    make_filename, build_manifest_row,
    load_processed_set, save_processed_set,
    load_json_config, save_json_config,
)

# Cross-age probability specific to test split (50%)
CROSS_AGE_P = 0.5

# Each kind maps to its list of engines; setting_name IS the kind (vc/tts/replay)
SETTINGS = {
    'vc'    : ['koko_vc',   'seed_vc',      'openvoice_vc'  ],
    'tts'   : ['koko_tts',  'xttsv2_tts',   'chatterbox_tts'],
    'replay': ['replay_c1', 'replay_c2',    'replay_c3'     ],
}

# Maps each kind to its dedicated output subdirectory name
KIND_DIR = {
    'vc'    : 'test_spoof_vc_clean',
    'tts'   : 'test_spoof_tts_clean',
    'replay': 'test_spoof_replay_clean',
}

# Module-level globals — populated by init()
OUT_BASE       = None  # Root directory for all generated test audio files
MANIFEST_BASE  = None  # Directory for manifest CSV files
JSON_BASE      = None  # Directory for JSON engine config files
PROCESSED_BASE = None  # Directory for processed-segment tracking files
SPOOFING_PATH  = None  # Path to the Spoofing package directory
SPOOFING_CORE  = None  # Path to the Spoofing/core directory
SPLIT_DATA     = None  # Dict holding loaded DataFrames and transcript lookup
n_total        = 0     # Total number of source segments in the test set
n_tts_valid    = 0     # Number of source segments that have a valid transcript (TTS only)


# =================================================================
# PATH HELPERS — test-specific naming conventions
# =================================================================

# Returns the manifest CSV path for a given test setting (vc/tts/replay).
def get_manifest_path(split, setting):
    return os.path.join(MANIFEST_BASE, f'test_spoof_{setting}_clean.csv')

# Returns the audio output directory for a given test setting and engine,
# using KIND_DIR because setting_name IS the kind in test.
def get_out_dir(split, setting, engine):
    return os.path.join(OUT_BASE, KIND_DIR[setting], engine)

# Returns the processed-segments JSON path for a given test setting.
def get_processed_path(split, setting):
    return os.path.join(PROCESSED_BASE, f'test_{setting}_processed.json')

# Returns the JSON config file path for a given test setting.
def get_json_config_path(split, setting):
    return os.path.join(JSON_BASE, f'test_{setting}_config.json')


# =================================================================
# INIT
# =================================================================

# Initialises all module-level globals, loads test CSVs from Drive,
# pre-creates output directories, and builds JSON configs if missing.
# n_tts_valid is computed separately because TTS pool is smaller
# (only segments that have a transcript are eligible).
def init(source_csv, target_csv, transcript_csv,
         out_base, manifest_base, json_base, processed_base,
         repo_dir):

    global OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE
    global SPOOFING_PATH, SPOOFING_CORE
    global SPLIT_DATA, n_total, n_tts_valid

    OUT_BASE       = out_base
    MANIFEST_BASE  = manifest_base
    JSON_BASE      = json_base
    PROCESSED_BASE = processed_base

    # Create all base directories
    for d in [OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE]:
        os.makedirs(d, exist_ok=True)

    # Pre-create one output folder per kind/engine combination
    for kind, engines in SETTINGS.items():
        for eng in engines:
            os.makedirs(os.path.join(OUT_BASE, KIND_DIR[kind], eng), exist_ok=True)

    SPOOFING_PATH = f'{repo_dir}/Spoofing'
    SPOOFING_CORE = f'{repo_dir}/Spoofing/core'

    set_seed(SEED)

    # Verify CSV paths exist on Drive before loading
    print('Path check:')
    for label, p in [('source', source_csv), ('target', target_csv), ('transcript', transcript_csv)]:
        ok = 'OK' if os.path.exists(p) else 'NOT FOUND'
        print(f'  [{ok}] {label}: {p}')

    print(f'\nOUT_BASE:       {OUT_BASE}')
    print(f'MANIFEST_BASE:  {MANIFEST_BASE}')
    print(f'JSON_BASE:      {JSON_BASE}')
    print(f'PROCESSED_BASE: {PROCESSED_BASE}')
    print(f'CROSS_AGE_P={CROSS_AGE_P}, TARGET_DUR>={TARGET_DUR}s, TARGET_SEC={TARGET_SEC}s, MIN_OUTPUT_SEC={MIN_OUTPUT_SEC}s')

    test_src = pd.read_csv(source_csv,     dtype={'speaker_id': str})
    test_tgt = pd.read_csv(target_csv,     dtype={'speaker_id': str})
    test_tr  = pd.read_csv(transcript_csv, dtype={'segment_id': str})

    # Build segment_id -> transcript lookup from rows that have a valid transcript
    test_tr_lookup = dict(zip(
        test_tr[test_tr['has_transcript'] == 1]['segment_id'].astype(str),
        test_tr[test_tr['has_transcript'] == 1]['sentence_clean']
    ))

    n_total     = len(test_src)
    # Count how many source segments actually have a transcript (used for TTS config)
    n_tts_valid = int(test_src['segment_id'].astype(str).isin(test_tr_lookup).sum())

    print(f'\nsource total    : {n_total}')
    print(f'with transcript : {n_tts_valid}  <- TTS only')
    print(f'target files    : {len(test_tgt)}')
    print(f'\nAge distribution:\n{test_src["mapped_age_class"].value_counts().to_string()}')

    SPLIT_DATA = {
        'test': {'src': test_src, 'tgt': test_tgt, 'tr': test_tr_lookup},
    }

    # Build JSON configs for all kinds; TTS uses n_tts_valid, others use n_total
    print('\nBuilding JSON configs...')
    for kind, engines in SETTINGS.items():
        n_segs = n_tts_valid if kind == 'tts' else n_total
        _build_json_config(kind, engines, n_segs, CROSS_AGE_P)

    print('\nAll JSON configs ready')


# =================================================================
# JSON CONFIG BUILDER
# =================================================================

# Builds and saves the per-engine JSON config for one test kind (vc/tts/replay).
# Divides n_segs evenly across the engines in the kind.
# replay engines get no cross_age fields.
# Skips creation if the config file already exists on disk.
ddef _build_json_config(kind, engines, n_segs, cross_age_p):
    json_path = os.path.join(JSON_BASE, f'test_{kind}_config.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f'  Config already exists for test/{kind}:')
        for eng, v in existing.items():
            ca  = f", cross_age={v['cross_age_count']}"        if 'cross_age_count' in v else ''
            cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
            print(f'    {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')
        return existing, json_path

    # Distributes total count as evenly as possible across n engines
    def split_evenly(total, n):
        base = total // n
        rem  = total % n
        return [base + (1 if i < rem else 0) for i in range(n)]

    counts = split_evenly(n_segs, len(engines))
    config = {}

    for eng, cnt in zip(engines, counts):
        config[eng] = {'count_required': cnt, 'count_done': 0}
        # Replay has no target speaker so no cross_age tracking needed
        if kind != 'replay':
            config[eng]['cross_age_count'] = round(cnt * cross_age_p)
            config[eng]['cross_age_done']  = 0

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'  New config created for test/{kind}')
    for eng, v in config.items():
        ca  = f", cross_age={v['cross_age_count']}"        if 'cross_age_count' in v else ''
        cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
        print(f'    {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')

    return config, json_path

# =================================================================
# STATUS CHECK
# =================================================================

# Prints a formatted progress bar summary for all test kinds and engines.
def status_check():
    print('\n' + '='*70)
    print('STATUS CHECK — TEST')
    print('='*70)
    for kind in SETTINGS:
        try:
            cfg = load_json_config(JSON_BASE, 'test', kind)
            print(f'\n[test/{kind}]')
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
            print(f'  Config not found for test/{kind} — run init() first')


# =================================================================
# MAIN ENGINE RUNNER
# =================================================================

# Runs one engine for the test split until count_required is reached.
#
# Flow:
#   1. Load JSON config; verify engine is present and not already complete.
#   2. Shuffle source pool with fixed SEED; exclude already-processed segments.
#      For TTS, further filter to segments that have a valid transcript.
#   3. Derive a deterministic per-engine RNG seed from (split, setting, engine).
#   4. Loop over available sources:
#        a. Determine cross_age flag based on remaining cross-age quota.
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
                src_row=src_row,
                tgt_row=tgt_row,
                kind=kind,
                eng=engine,
                cross_age=cross_age,
                spoofed_seg_path=out_path,
                tr_lookup=tr_lookup,
                split=split_name,
                setting=setting_name
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
    print(
        f'{split_name}/{setting_name}/{engine}: '
        f'done={ok_count}, total={final_done}/{count_required}, '
        f'cross_age={final_cross_done}/{cross_age_count}, '
        f'src_tried={src_tried}'
    )

    if final_done < count_required:
        print('  Source pool exhausted before reaching target count.')
    if kind != 'replay' and final_cross_done != cross_age_count:
        print(f'  cross_age mismatch: got {final_cross_done}, expected {cross_age_count}')
