import os, sys, math, random, shutil, gc, traceback, importlib, inspect, types, json
from datetime import datetime
import numpy as np, pandas as pd, librosa
import soundfile as sf
import torch
from tqdm.auto import tqdm
import importlib.util
import hashlib

import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE
        huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)
    except ImportError:
        huggingface_hub.is_offline_mode = lambda: False

from Spoofing.voice_conversion import run_vc_on_file
from Spoofing.text_to_speech   import run_tts_on_file
import Spoofing.replay_attack  as replay_module
from preprocessing import run_silero_vad

SEED         = 42
SR           = 16000
TARGET_SEC   = 3.0
TARGET_DUR   = 7.0
CROSS_AGE_P  = 0.8
MAX_TGT_TRIES = 20
MAX_SRC_TRIES = 50
TARGET_SR      = 16000
MERGE_GAP_MS   = 500
MIN_SPEECH_SEC = 3.0

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

KOKOCLONE_PATH = '/content/kokoclone'
KOKOCLONE_CORE = '/content/kokoclone/core'

REPLAY_CONFIG_FNS = {
    'replay_c1': replay_module.config1,
    'replay_c2': replay_module.config2,
    'replay_c3': replay_module.config3,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def engine_type(eng):
    if eng.endswith('_vc'):      return 'vc'
    if eng.endswith('_tts'):     return 'tts'
    if eng.startswith('replay_'): return 'replay'
    raise ValueError(f'Unknown engine: {eng}')


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


def save_audio(out_path, audio, sr=SR, target_sec=TARGET_SEC, src_sr=None):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if src_sr is not None and src_sr != sr:
        audio = librosa.resample(
            np.asarray(audio).squeeze().astype(np.float32),
            orig_sr=src_sr,
            target_sr=sr
        )
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


def setup_paths_for_engine(eng, spoofing_path, spoofing_core):
    for p in [KOKOCLONE_CORE, spoofing_core, KOKOCLONE_PATH, spoofing_path]:
        while p in sys.path: sys.path.remove(p)
    if eng in ('koko_tts', 'koko_vc'):
        sys.path.insert(0, KOKOCLONE_PATH); sys.path.insert(0, KOKOCLONE_CORE)
        for m in list(sys.modules):
            if m == 'core' or m.startswith('core.'): del sys.modules[m]
    elif eng == 'xttsv2_tts':
        sys.path.insert(0, spoofing_path); sys.path.insert(0, spoofing_core)
        for m in list(sys.modules):
            if m == 'core' or m.startswith('core.'): del sys.modules[m]
    else:
        sys.path.insert(0, KOKOCLONE_PATH); sys.path.insert(0, spoofing_path)


def run_engine(eng, src_path, tgt_path=None, text=None, spoofing_path=None, spoofing_core=None):
    setup_paths_for_engine(eng, spoofing_path, spoofing_core)
    if engine_type(eng) == 'vc':
        audio = run_vc_on_file(src_path, tgt_path, eng[:-3], sr=SR)
        return audio
    if engine_type(eng) == 'tts':
        ref = tgt_path
        if text is None or (isinstance(text, float) and math.isnan(text)):
            raise ValueError('Missing transcript')
        text = str(text).strip()
        if not text or text.lower() == 'nan':
            raise ValueError('Empty transcript')
        audio = run_tts_on_file(text=text, reference_audio_path=ref, model_name=eng[:-4])
        return audio
    if engine_type(eng) == 'replay':
        a, _ = librosa.load(src_path, sr=SR)
        return REPLAY_CONFIG_FNS[eng](a, sr=SR)
    raise ValueError(f'Unknown engine: {eng}')


def find_valid_target_file(tgt_df, speaker_id, rng, min_duration=TARGET_DUR):
    spk_files = tgt_df[
        (tgt_df['speaker_id'] == speaker_id) &
        (tgt_df['vad_status'] == 'success') &
        (tgt_df['speech_duration_sec'] >= min_duration)
    ]
    if len(spk_files) == 0:
        return None
    idx = rng.integers(0, len(spk_files))
    return spk_files.iloc[idx]['processed_path']


def pick_target(tgt_df, src_age, cross_age, rng, max_tries=MAX_TGT_TRIES):
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


def create_empty_manifest(p):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    if not os.path.exists(p):
        pd.DataFrame(columns=MANIFEST_COLUMNS).to_csv(p, index=False)


def make_filename(seg_id, eng, kind):
    return f'{kind}__{eng}__{seg_id}.wav'


def build_manifest_row(src_row, tgt_row, kind, eng, cross_age,
                        final_seg_path, tr_lookup, split, setting):
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
        'target_file_path'     : (np.nan if is_replay else safe_val(tgt_row.get('processed_path') if tgt_row is not None else np.nan)),
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


def remap(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].str.replace(
                "/content/drive/MyDrive/age verification/processed/data/spoof_targets",
                "/content/spoof_targets_extracted/spoof_targets",
                regex=False
            ).str.replace(
                "/content/drive/MyDrive/age verification/processed/data/real_candidates",
                "/content/real_clean_extracted/real_candidates",
                regex=False
            ).str.replace(
                "/content/drive/MyDrive/age verification/processed/data/cv_backup",
                "/content/real_clean_extracted/cv_backup",
                regex=False
            )
    return df
