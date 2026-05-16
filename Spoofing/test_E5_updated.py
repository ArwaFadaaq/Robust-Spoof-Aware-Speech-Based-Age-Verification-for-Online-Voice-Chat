# ============================================================
# E5 — TEST version (updated to match train/valid changes)
# ============================================================
# اللي اتغيّر:
#   1) cross_age counter في JSON بدل pre-shuffled flags
#   2) processed_path بدل file_path (في find_valid_target + manifest)
#   3) pick_target ما يعمل fallback لعمر مختلف — يرفع RuntimeError
#   4) TTS reference = tgt_path فقط (بدون or src_path)
#   5) cross_age_done tracking في الـ config + print حالته
# ============================================================

# =================================================================
# MANIFEST
# =================================================================
MANIFEST_COLUMNS = [
    'source_seg_id','parent_file_id','target_file_id',
    'start_sec','end_sec',
    'source_seg_path','source_file_path','target_file_path',
    'source_speaker_id','source_gender','source_dataset','target_dataset','source_age_class',
    'target_speaker_id','target_gender','target_age_class',
    'authenticity','spoof_type','spoof_engine',
    'cross_age_spoof','age_direction','source_transcript_id','source_transcript',
    'source_pool','target_pool','final_seg_path',
]

# Per-spoof-type paths (test specific)
def get_manifest_path(kind):
    return os.path.join(MANIFEST_BASE, f'test_spoof_{kind}_c_manifest.csv')

def get_out_dir(kind, engine):
    return os.path.join(OUT_BASE, f'test_spoof_{kind}_clean', engine)

def get_processed_path(kind):
    return os.path.join(PROCESSED_BASE, f'test_{kind}_processed.json')

def get_json_config_path(kind):
    return os.path.join(JSON_BASE, f'test_{kind}_config.json')

def create_empty_manifest(p):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    if not os.path.exists(p):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(p, index=False)

def load_processed_set(kind):
    p = get_processed_path(kind)
    if os.path.exists(p):
        with open(p) as f:
            return set(json.load(f))
    return set()

def save_processed_set(kind, processed_set):
    p = get_processed_path(kind)
    with open(p, 'w') as f:
        json.dump(list(processed_set), f)

def load_json_config(kind):
    p = get_json_config_path(kind)
    with open(p) as f:
        return json.load(f)

def save_json_config(kind, config):
    p = get_json_config_path(kind)
    with open(p, 'w') as f:
        json.dump(config, f, indent=2)

# =================================================================
# AUDIO HELPERS
# =================================================================
def derive_age_direction(s, t):
    def _short(v):
        if v is None or (isinstance(v, float) and math.isnan(v)): return None
        x = str(v).strip().lower()
        if x in ('minor', 'm'): return 'm'
        if x in ('adult', 'a'): return 'a'
        return None
    a, b = _short(s), _short(t)
    return f'{a}2{b}' if (a and b) else np.nan

def safe_val(v):
    if v is None: return np.nan
    if isinstance(v, float) and math.isnan(v): return np.nan
    return v

def safe_str(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    return str(v).strip().lower()

def pad_or_trim(audio, sr=SR, target_sec=TARGET_SEC):
    if hasattr(audio, 'detach'): audio = audio.detach().cpu().numpy()
    a = np.asarray(audio).squeeze().astype(np.float32)
    target_len = int(round(target_sec * sr))
    if len(a) > target_len: return a[:target_len]
    if len(a) < target_len: return np.pad(a, (0, target_len - len(a)), mode='constant')
    return a

def save_audio(out_path, audio, sr=SR, target_sec=TARGET_SEC):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    a = pad_or_trim(audio, sr=sr, target_sec=target_sec)
    sf.write(out_path, a, sr, subtype='PCM_16')
    return out_path

def extract_longest_voiced(waveform_np, sr=SR):
    t = torch.from_numpy(waveform_np).float()
    _, long_segs, _ = run_silero_vad(t, sr=sr)
    if not long_segs:
        return None, False
    start_s, end_s = max(long_segs, key=lambda x: x[1] - x[0])
    start_i = int(start_s * sr)
    end_i   = int(end_s   * sr)
    return waveform_np[start_i:end_i], True

# =================================================================
# ENGINE DISPATCHER
# =================================================================
KOKOCLONE_PATH = '/content/kokoclone'
KOKOCLONE_CORE = '/content/kokoclone/core'
SPOOFING_PATH  = f'{REPO_DIR}/Spoofing'
SPOOFING_CORE  = f'{REPO_DIR}/Spoofing/core'

REPLAY_CONFIG_FNS = {
    'replay_c1': replay_module.config1,
    'replay_c2': replay_module.config2,
    'replay_c3': replay_module.config3,
}

def setup_paths_for_engine(eng):
    for p in [KOKOCLONE_CORE, SPOOFING_CORE, KOKOCLONE_PATH, SPOOFING_PATH]:
        while p in sys.path: sys.path.remove(p)
    if eng in ('koko_tts', 'koko_vc'):
        sys.path.insert(0, KOKOCLONE_PATH); sys.path.insert(0, KOKOCLONE_CORE)
        for m in list(sys.modules):
            if m == 'core' or m.startswith('core.'): del sys.modules[m]
    elif eng == 'xttsv2_tts':
        sys.path.insert(0, SPOOFING_PATH); sys.path.insert(0, SPOOFING_CORE)
        for m in list(sys.modules):
            if m == 'core' or m.startswith('core.'): del sys.modules[m]
    else:
        sys.path.insert(0, KOKOCLONE_PATH); sys.path.insert(0, SPOOFING_PATH)

def run_engine(eng, src_path, tgt_path=None, text=None):
    setup_paths_for_engine(eng)
    if engine_type(eng) == 'vc':
        return run_vc_on_file(src_path, tgt_path, eng[:-3], sr=SR)
    if engine_type(eng) == 'tts':
        ref = tgt_path                                          # ← بدون fallback
        if text is None or (isinstance(text, float) and math.isnan(text)):
            raise ValueError('Missing transcript')
        text = str(text).strip()
        if not text or text.lower() == 'nan':
            raise ValueError('Empty transcript')
        return run_tts_on_file(text=text, reference_audio_path=ref, model_name=eng[:-4])
    if engine_type(eng) == 'replay':
        a, _ = librosa.load(src_path, sr=SR)
        return REPLAY_CONFIG_FNS[eng](a, sr=SR)
    raise ValueError(f'Unknown engine: {eng}')

# =================================================================
# TARGET SELECTION (no fallback to wrong age)
# =================================================================
def find_valid_target_file(tgt_df, speaker_id, rng, min_duration=TARGET_DUR):
    spk_files = tgt_df[
        (tgt_df['speaker_id'] == speaker_id) &
        (tgt_df['vad_status'] == 'success') &
        (tgt_df['speech_duration_sec'] >= min_duration)
    ]
    if len(spk_files) == 0:
        return None
    idx = rng.integers(0, len(spk_files))
    return spk_files.iloc[idx]['processed_path']             # ← processed_path

def pick_target(tgt_df, src_age, cross_age, rng, max_tries=MAX_TGT_TRIES):
    """
    Pick a target speaker:
      - cross_age=True  → different age class
      - cross_age=False → same age class
    Raises RuntimeError if no valid target found — no fallback to wrong age.
    """
    opposite_age = 'adult' if src_age == 'minor' else 'minor'
    desired_age  = opposite_age if cross_age else src_age

    age_pool = tgt_df[tgt_df['mapped_age_class'] == desired_age]['speaker_id'].unique().copy()
    rng.shuffle(age_pool)

    for spk in age_pool[:max_tries]:
        tgt_file = find_valid_target_file(tgt_df, spk, rng)
        if tgt_file and os.path.exists(tgt_file):
            tgt_row = tgt_df[tgt_df['speaker_id'] == spk].iloc[0]
            return tgt_row, tgt_file

    raise RuntimeError(
        f'No valid target found for cross_age={cross_age}, '
        f'src_age={src_age}, desired_age={desired_age}. '
        f'Check target CSV for vad_status=success and speech_duration_sec>={TARGET_DUR}.'
    )

# =================================================================
# MANIFEST WRITER
# =================================================================
class ManifestWriter:
    def __init__(self, csv_path, flush_every=10):
        self.csv_path = csv_path
        self.flush_every = flush_every
        self._buf = []
        if not os.path.exists(csv_path): create_empty_manifest(csv_path)

    def append(self, row):
        self._buf.append({c: row.get(c, np.nan) for c in MANIFEST_COLUMNS})
        if len(self._buf) >= self.flush_every: self.flush()

    def flush(self):
        if self._buf:
            pd.DataFrame(self._buf, columns=MANIFEST_COLUMNS).to_csv(
                self.csv_path, mode='a', header=False, index=False, na_rep='NaN')
            self._buf.clear()

    def close(self): self.flush()
    def __enter__(self): return self
    def __exit__(self, *a): self.close(); return False

def make_filename(seg_id, eng, kind):
    return f'{kind}__{eng}__{seg_id}.wav'

def build_manifest_row(src_row, tgt_row, kind, eng, cross_age,
                        final_seg_path, tr_lookup):
    is_replay = (kind == 'replay')
    src_age = safe_str(src_row.get('mapped_age_class'))
    tgt_age = (np.nan if is_replay else safe_str(
        tgt_row.get('mapped_age_class') if tgt_row is not None else np.nan))
    seg_id = str(src_row.get('segment_id', ''))
    return {
        'source_seg_id'        : safe_val(src_row.get('segment_id')),
        'parent_file_id'       : safe_val(src_row.get('parent_file_id')),
        'target_file_id'       : (np.nan if is_replay else safe_val(tgt_row.get('file_id') if tgt_row is not None else np.nan)),
        'start_sec'            : safe_val(src_row.get('start_sec')),
        'end_sec'              : safe_val(src_row.get('end_sec')),
        'source_seg_path'      : safe_val(src_row.get('seg_path')),
        'source_file_path'     : safe_val(src_row.get('source_file_path')),
        'target_file_path'     : (np.nan if is_replay else safe_val(tgt_row.get('processed_path') if tgt_row is not None else np.nan)),   # ← processed_path
        'source_speaker_id'    : safe_val(src_row.get('speaker_id')),
        'source_gender'        : safe_val(src_row.get('gender')),
        'source_dataset'       : safe_val(src_row.get('dataset_source')),
        'target_dataset'       : (np.nan if is_replay else safe_val(tgt_row.get('dataset') if tgt_row is not None else np.nan)),
        'source_age_class'     : src_age,
        'target_speaker_id'    : (np.nan if is_replay else safe_val(tgt_row.get('speaker_id') if tgt_row is not None else np.nan)),
        'target_gender'        : (np.nan if is_replay else safe_val(tgt_row.get('gender') if tgt_row is not None else np.nan)),
        'target_age_class'     : tgt_age,
        'authenticity'         : 'spoof',
        'spoof_type'           : kind,
        'spoof_engine'         : eng,
        'cross_age_spoof'      : (False if is_replay else bool(cross_age)),
        'age_direction'        : (np.nan if is_replay else derive_age_direction(src_age, tgt_age)),
        'source_transcript_id' : (safe_val(src_row.get('parent_file_id')) if kind == 'tts' else np.nan),
        'source_transcript'    : (safe_val(tr_lookup.get(seg_id)) if kind == 'tts' else np.nan),
        'source_pool'          : safe_val(src_row.get('pool')),
        'target_pool'          : (np.nan if is_replay else safe_val(tgt_row.get('pool') if tgt_row is not None else np.nan)),
        'final_seg_path'       : final_seg_path,
    }

# =================================================================
# MAIN: run_one_test_engine
# =================================================================
def run_one_test_engine(engine):
    """
    Run one engine on TEST split until count_required is reached.

    cross_age logic (NEW):
      - cross_age_count : total cross_age required (from JSON, never changes)
      - cross_age_done  : how many cross_age succeeded so far (tracked in JSON)
      - Per segment: if cross_age_done < cross_age_count → True, else False
      - Exact count guaranteed regardless of skips or resume.
      - Audio content unbiased because target selection uses rng_target.
    """
    kind = engine_type(engine)
    src_df    = SPLIT_DATA['test']['src'].copy()
    tgt_df    = SPLIT_DATA['test']['tgt'].copy()
    tr_lookup = SPLIT_DATA['test']['tr']

    # --- Load state ---
    config = load_json_config(kind)
    if engine not in config:
        print(f'⚠️  {engine} not in JSON config for test/{kind}')
        return

    eng_cfg         = config[engine]
    count_required  = eng_cfg['count_required']
    count_done      = eng_cfg['count_done']
    cross_age_count = eng_cfg.get('cross_age_count', 0)
    cross_age_done  = eng_cfg.get('cross_age_done', 0)

    if count_done >= count_required:
        print(f'🎉 test/{kind}/{engine}: already complete ({count_done}/{count_required})')
        return

    remaining     = count_required - count_done
    processed_set = load_processed_set(kind)

    manifest_path = get_manifest_path(kind)
    out_dir       = get_out_dir(kind, engine)
    os.makedirs(out_dir, exist_ok=True)
    create_empty_manifest(manifest_path)

    print(f'\n========== test/{kind}/{engine} | need={remaining}/{count_required} ==========')
    if kind != 'replay':
        print(f'   🎲 cross_age: {cross_age_done}/{cross_age_count} done so far')

    # Shuffle FULL df with fixed seed THEN filter processed
    all_src_shuffled = src_df.sample(frac=1, random_state=SEED)
    available_src    = all_src_shuffled[
        ~all_src_shuffled['segment_id'].astype(str).isin(processed_set)
    ].reset_index(drop=True)

    if kind == 'tts':
        valid_ids     = set(tr_lookup.keys())
        available_src = available_src[
            available_src['segment_id'].astype(str).isin(valid_ids)
        ].reset_index(drop=True)

    # Per-engine rng (deterministic + diverse)
    seed_str    = f'test_{kind}_{engine}'
    engine_seed = SEED + int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng_target  = np.random.default_rng(engine_seed)

    src_tried     = 0
    src_idx       = 0
    ok_count      = 0       # successes this run
    ok_cross      = 0       # cross_age successes this run
    batch_counter = 0

    pbar = tqdm(total=remaining, desc=f'test/{kind}/{engine}')

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

            # ---------- Determine cross_age (counter-based) ----------
            if kind == 'replay':
                cross_age = False
            else:
                cross_age = (cross_age_done + ok_cross) < cross_age_count

            # ---------- Pick target (no wrong-age fallback) ----------
            tgt_row, tgt_path = None, None
            if kind != 'replay':
                try:
                    tgt_row, tgt_path = pick_target(tgt_df, src_age, cross_age, rng=rng_target)
                except RuntimeError as e:
                    print(f'  ⚠️ [{engine}] {seg_id}: no valid target — {e}')
                    continue

            # ---------- Transcript ----------
            text = None
            if kind == 'tts':
                text = tr_lookup.get(seg_id)
                if not text or (isinstance(text, float) and math.isnan(text)):
                    continue

            # ---------- Run engine ----------
            out_filename = make_filename(seg_id, engine, kind)
            out_path     = os.path.join(out_dir, out_filename)

            try:
                raw_audio = run_engine(engine, src_path, tgt_path, text)
            except Exception as e:
                print(f'  FAIL [{engine}] src={seg_id} → {type(e).__name__}: {e}')
                continue

            if raw_audio is None:
                continue

            # ---------- Post-process ----------
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

            # ---------- Save ----------
            try:
                save_audio(out_path, final_audio, sr=SR, target_sec=TARGET_SEC)
            except Exception as e:
                print(f'  SAVE FAIL [{engine}] {seg_id} → {e}')
                continue

            # =========================================================
            # SUCCESS UPDATE
            # =========================================================
            ok_count      += 1
            if cross_age:
                ok_cross  += 1
            processed_set.add(seg_id)
            batch_counter += 1
            print(f'✅ [{engine}] saved: {seg_id} | cross_age={cross_age}')

            # Update in-memory config
            config[engine]['count_done']     = count_done + ok_count
            config[engine]['cross_age_done'] = cross_age_done + ok_cross

            if batch_counter % 10 == 0:
                save_json_config(kind, config)
                save_processed_set(kind, processed_set)

            mw.append(build_manifest_row(
                src_row=src_row, tgt_row=tgt_row, kind=kind, eng=engine,
                cross_age=cross_age, final_seg_path=out_path,
                tr_lookup=tr_lookup
            ))

            pbar.update(1)

    pbar.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    save_json_config(kind, config)
    save_processed_set(kind, processed_set)

    final_done       = count_done + ok_count
    final_cross_done = cross_age_done + ok_cross
    print(f'✅ test/{kind}/{engine}: done={ok_count}, total={final_done}/{count_required}, '
          f'cross_age={final_cross_done}/{cross_age_count}, src_tried={src_tried}')

    if final_done < count_required:
        print('  ⚠️ Source pool exhausted before reaching target count.')
    if kind != 'replay' and final_cross_done != cross_age_count:
        print(f'  ⚠️ cross_age mismatch: got {final_cross_done}, expected {cross_age_count}')

print('✅ run_one_test_engine ready')
