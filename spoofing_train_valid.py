import os, sys, gc, math, json, hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from Spoofing.spoofing_utils import (
    SEED, SR, TARGET_SEC, TARGET_DUR, CROSS_AGE_P, MAX_TGT_TRIES,
    REPLAY_CONFIG_FNS, MANIFEST_COLUMNS,
    set_seed, engine_type, remap,
    safe_val, safe_str,
    save_audio, extract_longest_voiced,
    run_engine, find_valid_target_file, pick_target,
    ManifestWriter, create_empty_manifest,
    make_filename, build_manifest_row,
)

set_seed(SEED)

SETTINGS = {
    'set1': ['openvoice_vc', 'koko_vc',      'koko_tts',       'xttsv2_tts',        'replay_c1', 'replay_c2'],
    'set2': ['seed_vc',      'koko_vc',      'chatterbox_tts', 'xttsv2_tts',        'replay_c3', 'replay_c2'],
    'set3': ['seed_vc',      'openvoice_vc', 'koko_tts',       'chatterbox_tts', 'replay_c1', 'replay_c3'],
}


def init(project_root, repo_dir):
    global SPLIT_PATHS, OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE
    global SPOOFING_PATH, SPOOFING_CORE, SPLIT_DATA

    SPLIT_PATHS = {
        'train': {
            'source'    : f'{project_root}/processed/manifest/real_clean_splits/final_split/train_spoof_source_clean.csv',
            'target'    : f'{project_root}/processed/manifest/spoof_targets_splits/train_spoof_targets.csv',
            'transcript': f'{project_root}/spoofing/transcripts/train_spoof_c_transcript_inventory.csv',
        },
        'valid': {
            'source'    : f'{project_root}/processed/manifest/real_clean_splits/final_split/valid_spoof_source_clean.csv',
            'target'    : f'{project_root}/processed/manifest/spoof_targets_splits/val_spoof_targets.csv',
            'transcript': f'{project_root}/spoofing/transcripts/valid_spoof_c_transcript_inventory.csv',
        },
    }

    OUT_BASE       = f'{project_root}/spoofing/data'
    MANIFEST_BASE  = f'{project_root}/spoofing/manifest'
    JSON_BASE      = f'{project_root}/spoofing/intermediate_data/json_configs'
    PROCESSED_BASE = f'{project_root}/spoofing/intermediate_data/processed_segments'

    for d in [OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE]:
        os.makedirs(d, exist_ok=True)

    for split in ['train', 'valid']:
        for setting_name, engines in SETTINGS.items():
            for eng in engines:
                os.makedirs(os.path.join(OUT_BASE, split, setting_name, eng), exist_ok=True)

    SPOOFING_PATH = f'{repo_dir}/Spoofing'
    SPOOFING_CORE = f'{repo_dir}/Spoofing/core'

    print(f'\n✅ OUT_BASE:       {OUT_BASE}')
    print(f'✅ MANIFEST_BASE:  {MANIFEST_BASE}')
    print(f'✅ JSON_BASE:      {JSON_BASE}')
    print(f'✅ PROCESSED_BASE: {PROCESSED_BASE}')
    print(f'✅ CROSS_AGE_P={CROSS_AGE_P}, TARGET_DUR≥{TARGET_DUR}s, TARGET_SEC={TARGET_SEC}s')

    train_src, train_tgt, train_tr_lookup = _load_split_data('train')
    valid_src, valid_tgt, valid_tr_lookup = _load_split_data('valid')

    print(f'📊 TRAIN: {len(train_src)} source segs, {len(train_tgt)} target files, {len(train_tr_lookup)} transcripts')
    print(f'📊 VALID: {len(valid_src)} source segs, {len(valid_tgt)} target files, {len(valid_tr_lookup)} transcripts')
    print(f'\n🎯 TRAIN age distribution:\n{train_src["mapped_age_class"].value_counts().to_string()}')
    print(f'\n🎯 VALID age distribution:\n{valid_src["mapped_age_class"].value_counts().to_string()}')

    SRC_COLS = ['seg_path', 'source_file_path']
    TGT_COLS = ['processed_path']

    train_src = remap(train_src, SRC_COLS)
    train_tgt = remap(train_tgt, TGT_COLS)
    valid_src = remap(valid_src, SRC_COLS)
    valid_tgt = remap(valid_tgt, TGT_COLS)

    sample_src = train_src['seg_path'].dropna().iloc[0]
    sample_tgt = train_tgt['processed_path'].dropna().iloc[0]
    print(f'  src sample : {sample_src}')
    print(f'  src exists : {os.path.exists(sample_src)}')
    print(f'  tgt sample : {sample_tgt}')
    print(f'  tgt exists : {os.path.exists(sample_tgt)}')
    print('✅ Paths remapped to local')

    SPLIT_DATA = {
        'train': {'src': train_src, 'tgt': train_tgt, 'tr': train_tr_lookup},
        'valid': {'src': valid_src, 'tgt': valid_tgt, 'tr': valid_tr_lookup},
    }

    _build_json_configs_if_missing()


def _load_split_data(split):
    src = pd.read_csv(SPLIT_PATHS[split]['source'],     dtype={'speaker_id': str})
    tgt = pd.read_csv(SPLIT_PATHS[split]['target'],     dtype={'speaker_id': str})
    tr  = pd.read_csv(SPLIT_PATHS[split]['transcript'], dtype={'segment_id': str})
    lookup = dict(zip(
        tr[tr['has_transcript'] == 1]['segment_id'].astype(str),
        tr[tr['has_transcript'] == 1]['sentence_clean']
    ))
    return src, tgt, lookup


def get_manifest_path(split, setting):
    return os.path.join(MANIFEST_BASE, f'{split}_spoof_{setting}_clean.csv')

def get_out_dir(split, setting, engine):
    return os.path.join(OUT_BASE, split, setting, engine)

def get_processed_path(split, setting):
    return os.path.join(PROCESSED_BASE, f'{split}_{setting}_processed.json')

def get_json_config_path(split, setting):
    return os.path.join(JSON_BASE, f'{split}_{setting}_config.json')

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


def _build_json_config(split, setting_name, engines, n_src, cross_age_p):
    json_path = get_json_config_path(split, setting_name)

    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f'   ⏩ Config already exists for {split}/{setting_name} — skipping rebuild')
        return existing, json_path

    vc_engines     = [e for e in engines if engine_type(e) == 'vc']
    tts_engines    = [e for e in engines if engine_type(e) == 'tts']
    replay_engines = [e for e in engines if engine_type(e) == 'replay']

    n_per_type = n_src // 3
    remainder  = n_src - 3 * n_per_type

    n_vc     = n_per_type + (1 if remainder > 0 else 0)
    n_tts    = n_per_type + (1 if remainder > 1 else 0)
    n_replay = n_src - n_vc - n_tts

    def split_evenly(total, n):
        base = total // n
        rem  = total % n
        return [base + (1 if i < rem else 0) for i in range(n)]

    vc_counts     = split_evenly(n_vc, len(vc_engines)) if vc_engines else []
    tts_counts    = split_evenly(n_tts, len(tts_engines)) if tts_engines else []
    replay_counts = split_evenly(n_replay, len(replay_engines)) if replay_engines else []

    config = {}

    for eng, cnt in zip(vc_engines, vc_counts):
        config[eng] = {
            'count_required'   : cnt,
            'cross_age_count'  : round(cnt * cross_age_p),
            'cross_age_done'   : 0,
            'count_done'       : 0,
        }

    for eng, cnt in zip(tts_engines, tts_counts):
        config[eng] = {
            'count_required'   : cnt,
            'cross_age_count'  : round(cnt * cross_age_p),
            'cross_age_done'   : 0,
            'count_done'       : 0,
        }

    for eng, cnt in zip(replay_engines, replay_counts):
        config[eng] = {
            'count_required'   : cnt,
            'count_done'       : 0,
        }

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'   ✅ New config created for {split}/{setting_name}')
    return config, json_path


def _build_json_configs_if_missing():
    print('\n📐 Building JSON configs...')
    for split in ['train', 'valid']:
        n_src = len(SPLIT_DATA[split]['src'])
        print(f'\n  [{split}] n_src={n_src}')
        for setting_name, engines in SETTINGS.items():
            cfg, path = _build_json_config(split, setting_name, engines, n_src, CROSS_AGE_P)
            print(f'    {setting_name}: {path}')
            for eng, v in cfg.items():
                ca  = f", cross_age={v['cross_age_count']}"       if 'cross_age_count' in v else ''
                cad = f", cross_done={v.get('cross_age_done', 0)}" if 'cross_age_count' in v else ''
                print(f'      {eng}: required={v["count_required"]}{ca}{cad}, done={v["count_done"]}')
    print('\n✅ All JSON configs ready')


def status_check():
    print('\n' + '='*70)
    print('STATUS CHECK')
    print('='*70)
    for split in ['train', 'valid']:
        for setting_name in ['set1', 'set2', 'set3']:
            try:
                cfg = load_json_config(split, setting_name)
                print(f'\n[{split}/{setting_name}]')
                total_req = total_done = 0
                for eng, v in cfg.items():
                    req  = v['count_required']
                    done = v['count_done']
                    pct  = 100 * done / req if req > 0 else 0
                    ca   = f" cross_age={v['cross_age_count']}" if 'cross_age_count' in v else ''
                    bar  = '█' * int(pct // 10) + '░' * (10 - int(pct // 10))
                    status = '✅' if done >= req else '🔄'
                    print(f'  {status} {eng:<18} [{bar}] {done:>5}/{req:<5} ({pct:5.1f}%){ca}')
                    total_req  += req
                    total_done += done
                pct_total = 100 * total_done / total_req if total_req > 0 else 0
                print(f'  TOTAL: {total_done}/{total_req} ({pct_total:.1f}%)')
            except FileNotFoundError:
                print(f'  ❌ Config not found for {split}/{setting_name}')


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

            config[engine]['count_done']     = count_done + ok_count
            config[engine]['cross_age_done'] = cross_age_done + ok_cross

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


def run_one_engine_chatterbox(split_name, setting_name, engine='chatterbox_tts'):

    kind      = 'tts'
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
    print(f'   🎲 cross_age: {cross_age_done}/{cross_age_count} done so far')

    all_src_shuffled = src_df.sample(frac=1, random_state=SEED)
    available_src    = all_src_shuffled[
        ~all_src_shuffled['segment_id'].astype(str).isin(processed_set)
    ].reset_index(drop=True)

    valid_ids     = set(tr_lookup.keys())
    available_src = available_src[
        available_src['segment_id'].astype(str).isin(valid_ids)
    ].reset_index(drop=True)

    seed_str    = f'{split_name}_{setting_name}_{engine}'
    engine_seed = SEED + int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng_target  = np.random.default_rng(engine_seed)

    src_tried = 0
    src_idx   = 0
    ok_count  = 0
    ok_cross  = 0

    pbar = tqdm(total=remaining, desc=f'{split_name}/{setting_name}/{engine}')

    with ManifestWriter(manifest_path, flush_every=1) as mw:
        while ok_count < remaining and src_idx < len(available_src):

            src_row   = available_src.iloc[src_idx]
            src_idx  += 1
            src_tried += 1

            seg_id   = str(src_row.get('segment_id', ''))
            src_path = str(src_row.get('seg_path', ''))
            src_age  = safe_str(src_row.get('mapped_age_class'))

            if not src_path or not os.path.exists(src_path):
                continue

            cross_age   = (cross_age_done + ok_cross) < cross_age_count
            desired_age = ('adult' if src_age == 'minor' else 'minor') if cross_age else src_age

            text = tr_lookup.get(seg_id)
            if not text or (isinstance(text, float) and math.isnan(text)):
                continue
            text = str(text).strip()
            if not text or text.lower() == 'nan':
                continue

            age_pool = tgt_df[
                tgt_df['mapped_age_class'] == desired_age
            ]['speaker_id'].unique().copy()
            rng_target.shuffle(age_pool)

            saved = False

            for spk in age_pool[:50]:

                tgt_file = find_valid_target_file(tgt_df, spk, rng_target)
                if not tgt_file or not os.path.exists(tgt_file):
                    continue

                tgt_row      = tgt_df[tgt_df['speaker_id'] == spk].iloc[0]
                out_filename = make_filename(seg_id, engine, kind)
                out_path     = os.path.join(out_dir, out_filename)

                try:
                    raw_audio = run_engine(engine, src_path, tgt_file, text,
                                           spoofing_path=SPOOFING_PATH, spoofing_core=SPOOFING_CORE)
                except Exception as e:
                    print(f'  FAIL [{engine}] src={seg_id} tgt={spk} → {type(e).__name__}: {e}')
                    continue

                if raw_audio is None:
                    continue

                if hasattr(raw_audio, 'detach'):
                    raw_np = raw_audio.detach().cpu().numpy()
                else:
                    raw_np = np.asarray(raw_audio).squeeze().astype(np.float32)

                voiced, found = extract_longest_voiced(raw_np, sr=SR)

                if not found:
                    print(f'  ⚠️  [{engine}] VAD failed with target={spk} — trying next target...')
                    continue

                try:
                    save_audio(out_path, voiced, sr=SR, target_sec=TARGET_SEC)
                except Exception as e:
                    print(f'  SAVE FAIL [{engine}] {seg_id} → {e}')
                    continue

                saved          = True
                ok_count      += 1
                if cross_age:
                    ok_cross  += 1
                processed_set.add(seg_id)

                print(f'✅ [{engine}] saved: {seg_id} | tgt={spk} | cross_age={cross_age}')

                config[engine]['count_done']     = count_done + ok_count
                config[engine]['cross_age_done'] = cross_age_done + ok_cross

                save_json_config(split_name, setting_name, config)
                save_processed_set(split_name, setting_name, processed_set)

                mw.append(build_manifest_row(
                    src_row=src_row, tgt_row=tgt_row, kind=kind, eng=engine,
                    cross_age=cross_age, final_seg_path=out_path,
                    tr_lookup=tr_lookup, split=split_name, setting=setting_name
                ))

                pbar.update(1)
                break

            if not saved:
                print(f'  ❌ [{engine}] {seg_id}: all 50 target candidates exhausted — segment skipped')

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
    if final_cross_done != cross_age_count:
        print(f'  ⚠️ cross_age mismatch: got {final_cross_done}, expected {cross_age_count}')
