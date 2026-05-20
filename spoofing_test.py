import os, sys, gc, math, json, hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from Spoofing.spoofing_utils import (
    SEED, SR, TARGET_SEC, TARGET_DUR, MAX_TGT_TRIES,
    REPLAY_CONFIG_FNS, MANIFEST_COLUMNS,
    set_seed, engine_type, remap,
    safe_val, safe_str,
    save_audio, extract_longest_voiced,
    run_engine, find_valid_target_file, pick_target,
    ManifestWriter, create_empty_manifest,
    make_filename, build_manifest_row,
)

CROSS_AGE_P   = 0.5
MIN_SPEECH_SEC = 3.0
TARGET_SR      = 16000
MERGE_GAP_MS   = 500
MAX_SRC_TRIES  = 50

set_seed(SEED)

SETTINGS = {
    'vc'    : ['koko_vc',    'seed_vc',       'openvoice_vc'  ],
    'tts'   : ['koko_tts',   'xttsv2_tts',    'chatterbox_tts'],
    'replay': ['replay_c1',  'replay_c2',     'replay_c3'     ],
}

KIND_DIR = {
    'vc'    : 'test_spoof_vc_clean',
    'tts'   : 'test_spoof_tts_clean',
    'replay': 'test_spoof_replay_clean',
}

SOURCE_CSV     = ''
TARGET_CSV     = ''
TRANSCRIPT_CSV = ''
OUT_BASE       = ''
MANIFEST_BASE  = ''
JSON_BASE      = ''
PROCESSED_BASE = ''
SPOOFING_PATH  = ''
SPOOFING_CORE  = ''
SPLIT_DATA     = {}
n_total        = 0
n_tts_valid    = 0


def init(project_root, repo_dir):
    global SOURCE_CSV, TARGET_CSV, TRANSCRIPT_CSV
    global OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE
    global SPOOFING_PATH, SPOOFING_CORE
    global SPLIT_DATA, n_total, n_tts_valid

    SOURCE_CSV     = f'{project_root}/processed/manifest/real_clean_splits/final_split/test_spoof_source_clean.csv'
    TARGET_CSV     = f'{project_root}/processed/manifest/spoof_targets_splits/test_spoof_targets.csv'
    TRANSCRIPT_CSV = f'{project_root}/spoofing/transcripts/test_spoof_c_transcript_inventory.csv'

    OUT_BASE       = f'{project_root}/spoofing/data/test'
    MANIFEST_BASE  = '/content/drive/MyDrive/age verification/spoofing/manifest'
    JSON_BASE      = f'{project_root}/spoofing/intermediate_data/json_configs'
    PROCESSED_BASE = f'{project_root}/spoofing/intermediate_data/processed_segments'

    for d in [OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE]:
        os.makedirs(d, exist_ok=True)

    for kind, engines in SETTINGS.items():
        for eng in engines:
            os.makedirs(os.path.join(OUT_BASE, KIND_DIR[kind], eng), exist_ok=True)

    SPOOFING_PATH = f'{repo_dir}/Spoofing'
    SPOOFING_CORE = f'{repo_dir}/Spoofing/core'

    print('\n🔍 Path check:')
    for label, p in [('source', SOURCE_CSV), ('target', TARGET_CSV), ('transcript', TRANSCRIPT_CSV)]:
        ok = '✅' if os.path.exists(p) else '❌'
        print(f'   {ok} {label}: {p}')
    print(f'\n✅ OUT_BASE:       {OUT_BASE}')
    print(f'✅ MANIFEST_BASE:  {MANIFEST_BASE}')
    print(f'✅ JSON_BASE:      {JSON_BASE}')
    print(f'✅ PROCESSED_BASE: {PROCESSED_BASE}')
    print(f'✅ CROSS_AGE_P={CROSS_AGE_P}, TARGET_DUR>={TARGET_DUR}s, TARGET_SEC={TARGET_SEC}s')

    test_src = pd.read_csv(SOURCE_CSV,     dtype={'speaker_id': str})
    test_tgt = pd.read_csv(TARGET_CSV,     dtype={'speaker_id': str})
    test_tr  = pd.read_csv(TRANSCRIPT_CSV, dtype={'segment_id': str})

    test_tr_lookup = dict(zip(
        test_tr[test_tr['has_transcript'] == 1]['segment_id'].astype(str),
        test_tr[test_tr['has_transcript'] == 1]['sentence_clean']
    ))

    n_total     = len(test_src)
    n_tts_valid = int(test_src['segment_id'].astype(str).isin(test_tr_lookup).sum())

    print(f'📊 source total    : {n_total}')
    print(f'📊 with transcript : {n_tts_valid}  <- TTS only')
    print(f'📊 target files    : {len(test_tgt)}')
    print(f'\n🎯 Age distribution:\n{test_src["mapped_age_class"].value_counts().to_string()}')

    test_src = remap(test_src, ['seg_path', 'source_file_path'])
    test_tgt = remap(test_tgt, ['processed_path'])

    sample_src = test_src['seg_path'].dropna().iloc[0]
    sample_tgt = test_tgt['processed_path'].dropna().iloc[0]
    print(f'\n  src sample : {sample_src}')
    print(f'  src exists : {os.path.exists(sample_src)}')
    print(f'  tgt sample : {sample_tgt}')
    print(f'  tgt exists : {os.path.exists(sample_tgt)}')
    print('✅ Paths remapped to local')

    SPLIT_DATA = {
        'test': {'src': test_src, 'tgt': test_tgt, 'tr': test_tr_lookup},
    }

    _build_json_configs_if_missing()


def get_manifest_path(split, setting):
    return os.path.join(MANIFEST_BASE, f'test_spoof_{setting}_clean.csv')

def get_out_dir(split, setting, engine):
    return os.path.join(OUT_BASE, KIND_DIR[setting], engine)

def get_processed_path(split, setting):
    return os.path.join(PROCESSED_BASE, f'test_{setting}_processed.json')

def get_json_config_path(split, setting):
    return os.path.join(JSON_BASE, f'test_{setting}_config.json')

def load_processed_set(split, setting):
    p = get_processed_path(split, setting)
    if os.path.exists(p):
        with open(p) as f:
            return set(json.load(f))
    return set()

def save_processed_set(split, setting, processed_set):
    p = get_processed_path(split, setting)
    with open(p, 'w') as f:
        json.dump(list(processed_set), f)

def load_json_config(split, setting):
    p = get_json_config_path(split, setting)
    with open(p) as f:
        return json.load(f)

def save_json_config(split, setting, config):
    p = get_json_config_path(split, setting)
    with open(p, 'w') as f:
        json.dump(config, f, indent=2)


def _build_json_config(kind, engines, n_segs, cross_age_p):
    json_path = os.path.join(JSON_BASE, f'test_{kind}_config.json')

    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f'   ⏩ Config already exists for test/{kind} — skipping rebuild')
        return existing, json_path

    def split_evenly(total, n):
        base = total // n
        rem  = total % n
        return [base + (1 if i < rem else 0) for i in range(n)]

    counts = split_evenly(n_segs, len(engines))
    config = {}

    for eng, cnt in zip(engines, counts):
        config[eng] = {'count_required': cnt, 'count_done': 0}
        if kind != 'replay':
            config[eng]['cross_age_count'] = round(cnt * cross_age_p)
            config[eng]['cross_age_done']  = 0

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'   ✅ New config created for test/{kind}')
    return config, json_path


def _build_json_configs_if_missing():
    print('\n📐 Building JSON configs...')
    for kind, engines in SETTINGS.items():
        n_segs = n_tts_valid if kind == 'tts' else n_total
        cfg, path = _build_json_config(kind, engines, n_segs, CROSS_AGE_P)
        label = 'transcript-filtered' if kind == 'tts' else 'full pool'
        print(f'\n  [test/{kind}]  n_segs={n_segs} ({label})')
        for eng, v in cfg.items():
            ca  = f", cross_age={v['cross_age_count']}"       if 'cross_age_count' in v else ''
            cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
            print(f'      {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')
    print('\n✅ All JSON configs ready')


def status_check():
    print('\n' + '='*70)
    print('STATUS CHECK — TEST')
    print('='*70)
    for kind, engines in SETTINGS.items():
        try:
            cfg = load_json_config('test', kind)
            print(f'\n[test/{kind}]')
            total_req = total_done = 0
            for eng, v in cfg.items():
                req  = v['count_required']
                done = v['count_done']
                pct  = 100 * done / req if req > 0 else 0
                ca   = f" cross_age={v['cross_age_count']}" if 'cross_age_count' in v else ''
                bar  = '█' * int(pct // 10) + '░' * (10 - int(pct // 10))
                st   = '✅' if done >= req else '🔄'
                print(f'  {st} {eng:<18} [{bar}] {done:>5}/{req:<5} ({pct:5.1f}%){ca}')
                total_req  += req
                total_done += done
            pct_total = 100 * total_done / total_req if total_req > 0 else 0
            print(f'  TOTAL: {total_done}/{total_req} ({pct_total:.1f}%)')
        except FileNotFoundError:
            print(f'  ❌ Config not found for test/{kind} — run init() first')


def run_one_engine(split_name, setting_name, engine):
    kind      = engine_type(engine)
    src_df    = SPLIT_DATA[split_name]['src'].copy()
    tgt_df    = SPLIT_DATA[split_name]['tgt'].copy()
    tr_lookup = SPLIT_DATA[split_name]['tr']

    config = load_json_config(split_name, setting_name)
    if engine not in config:
        print(f'⚠️  {engine} not in JSON config for {split_name}/{setting_name}')
        return

    eng_cfg         = config[engine]
    count_required  = eng_cfg['count_required']
    count_done      = eng_cfg['count_done']
    cross_age_count = eng_cfg.get('cross_age_count', 0)
    cross_age_done  = eng_cfg.get('cross_age_done', 0)

    if count_done >= count_required:
        print(f'🎉 {split_name}/{setting_name}/{engine}: already complete ({count_done}/{count_required})')
        return

    remaining     = count_required - count_done
    processed_set = load_processed_set(split_name, setting_name)
    manifest_path = get_manifest_path(split_name, setting_name)
    out_dir       = get_out_dir(split_name, setting_name, engine)
    os.makedirs(out_dir, exist_ok=True)
    create_empty_manifest(manifest_path)

    print(f'\n========== {split_name}/{setting_name} | {engine} | need={remaining}/{count_required} ==========')
    if kind != 'replay':
        print(f'   🎲 cross_age: {cross_age_done}/{cross_age_count} done so far')

    all_src_shuffled = src_df.sample(frac=1, random_state=SEED)
    available_src    = all_src_shuffled[
        ~all_src_shuffled['segment_id'].astype(str).isin(processed_set)
    ].reset_index(drop=True)

    if kind == 'tts':
        valid_ids     = set(tr_lookup.keys())
        available_src = available_src[
            available_src['segment_id'].astype(str).isin(valid_ids)
        ].reset_index(drop=True)

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

            if kind == 'replay':
                cross_age = False
            else:
                cross_age = (cross_age_done + ok_cross) < cross_age_count

            tgt_row  = None
            tgt_path = None
            if kind != 'replay':
                try:
                    tgt_row, tgt_path = pick_target(tgt_df, src_age, cross_age, rng=rng_target)
                except RuntimeError as e:
                    print(f'  ⚠️ [{engine}] {seg_id}: no valid target — {e}')
                    continue

            text = None
            if kind == 'tts':
                text = tr_lookup.get(seg_id)
                if not text or (isinstance(text, float) and math.isnan(text)):
                    continue

            out_filename = make_filename(seg_id, engine, kind)
            out_path     = os.path.join(out_dir, out_filename)

            try:
                raw_audio = run_engine(engine, src_path, tgt_path, text,
                                       spoofing_path=SPOOFING_PATH, spoofing_core=SPOOFING_CORE)
            except Exception as e:
                print(f'  FAIL [{engine}] src={seg_id} → {type(e).__name__}: {e}')
                continue

            if raw_audio is None:
                continue

            if kind == 'tts':
                if hasattr(raw_audio, 'detach'):
                    raw_np = raw_audio.detach().cpu().numpy()
                else:
                    raw_np = np.asarray(raw_audio).squeeze().astype(np.float32)
                voiced, found = extract_longest_voiced(raw_np, sr=SR)
                if not found:
                    continue
                final_audio = voiced
            else:
                if hasattr(raw_audio, 'detach'):
                    raw_np = raw_audio.detach().cpu().numpy()
                else:
                    raw_np = np.asarray(raw_audio).squeeze().astype(np.float32)
                final_audio = raw_np

            try:
                save_audio(out_path, final_audio, sr=SR, target_sec=TARGET_SEC)
            except Exception as e:
                print(f'  SAVE FAIL [{engine}] {seg_id} → {e}')
                continue

            ok_count      += 1
            if cross_age:
                ok_cross  += 1
            processed_set.add(seg_id)
            batch_counter += 1
            print(f'✅ [{engine}] saved: {seg_id} | cross_age={cross_age}')

            config[engine]['count_done']      = count_done + ok_count
            config[engine]['cross_age_done']  = cross_age_done + ok_cross

            if batch_counter % 10 == 0:
                save_json_config(split_name, setting_name, config)
                save_processed_set(split_name, setting_name, processed_set)

            mw.append(build_manifest_row(
                src_row=src_row, tgt_row=tgt_row, kind=kind, eng=engine,
                cross_age=cross_age, final_seg_path=out_path,
                tr_lookup=tr_lookup, split=split_name, setting=setting_name
            ))

            pbar.update(1)

    pbar.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_json_config(split_name, setting_name, config)
    save_processed_set(split_name, setting_name, processed_set)

    final_done       = count_done + ok_count
    final_cross_done = cross_age_done + ok_cross
    print(f'✅ {split_name}/{setting_name}/{engine}: done={ok_count}, total={final_done}/{count_required}, '
          f'cross_age={final_cross_done}/{cross_age_count}, src_tried={src_tried}')

    if final_done < count_required:
        print('  ⚠️ Source pool exhausted before reaching target count.')
    if kind != 'replay' and final_cross_done != cross_age_count:
        print(f'  ⚠️ cross_age mismatch: got {final_cross_done}, expected {cross_age_count}')
