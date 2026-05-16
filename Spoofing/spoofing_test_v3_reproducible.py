# -*- coding: utf-8 -*-
"""spoofing_test_v3_reproducible.ipynb

# 🎯 Spoofing — TEST Generation v3 (reproducible)

نفس فلسفة train_valid v3 (JSON configs، processed_set، rng deterministic،
shuffle قبل الفلترة، VAD للـ TTS، resume تلقائي) — بس للـ test split.

## التقسيمة:
- 3 أنواع spoof (vc / tts / replay) — كل نوع يأخذ الـ source كامل
- 3 موديلات لكل نوع (33/33/33 داخل النوع)
- cross_age = 0.5 (متوازنة للتقييم)
- 3 منفست منفصلين (per spoof type)

## 🟢 PHASE A — Mount + Clone
"""

# ---- A1: Mount Drive + Clone repo ----
import os, sys, glob
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

candidates  = ['/content/drive/MyDrive/age verification',
               '/content/drive/Shareddrives/age verification']
candidates += glob.glob('/content/drive/Shareddrives/*/age verification')
candidates += glob.glob('/content/drive/MyDrive/*/age verification')
PROJECT_ROOT = next((p for p in candidates if os.path.isdir(p)), None)
assert PROJECT_ROOT, 'Could not find "age verification" folder.'
print('PROJECT_ROOT =', PROJECT_ROOT)

GITHUB_PAT  = 'ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'   # ← التوكن الجديد
GITHUB_USER = 'ArwaFadaaq'
REPO_NAME   = 'Robust-Multimodal-Age-Verification-for-Online-Gaming-Chat'
REPO_DIR    = f'/content/{REPO_NAME}'

if not os.path.isdir(REPO_DIR):
    os.system(f'git clone https://{GITHUB_PAT}@github.com/{GITHUB_USER}/{REPO_NAME}.git {REPO_DIR}')
    print('Cloned repo →', REPO_DIR)
else:
    os.system(f'cd {REPO_DIR} && git pull')
    print('Repo updated at', REPO_DIR)

for p in (REPO_DIR, f'{REPO_DIR}/Spoofing',
          '/content/kokoclone', '/content/kokoclone/core',
          '/content/OpenVoice', '/content/seed-vc'):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

print('A1 done.')


"""## 🟡 PHASE B — TTS installs → Restart Runtime"""

# ---- B1: koko-tts ----
import os, sys
os.chdir('/content')
!pip install --upgrade pip -q
!pip install gradio_client -q

if not os.path.isdir('/content/kokoclone'):
    !git clone https://github.com/Ashish-Patnaik/kokoclone.git

!pip install -r /content/kokoclone/requirements.txt -q
!pip install kokoro-onnx[gpu] -q
!pip install pydub -q

if '/content/kokoclone' not in sys.path:
    sys.path.insert(0, '/content/kokoclone')
print('koko-tts installed.')

# ---- B2: xtts v2 ----
!pip install -q coqui-tts

import os
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
from TTS.api import TTS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
print('xtts-v2 installed.')

# ---- B3: chatterbox-tts ----
!pip uninstall -y chatterbox-tts resemble-perth torchvision -q
!pip install torch==2.5.0 torchaudio==2.5.0 -q
!pip install transformers==4.46.3 diffusers==0.29.0 -q
!pip install huggingface_hub accelerate librosa==0.11.0 safetensors soundfile scipy -q
!pip install resemble-perth s3tokenizer conformer -q
!pip install chatterbox-tts --no-deps -q
print('chatterbox-tts installed.')
print('\n>>> RESTART RUNTIME NOW, then run PHASE C. <<<')


"""## 🟡 PHASE C — VC installs → Restart Runtime"""

# ---- C1: seed-vc ----
import os, sys
os.chdir('/content')
if not os.path.isdir('/content/seed-vc'):
    !git clone https://github.com/Plachtaa/seed-vc.git
!pip install -r /content/seed-vc/requirements.txt -q
if '/content/seed-vc' not in sys.path:
    sys.path.insert(0, '/content/seed-vc')
print('seed-vc installed.')

# ---- C2: openvoice v2 ----
import os, sys
os.chdir('/content')

!pip install numpy==1.26.4 -q
!pip install torch==2.1.0 torchaudio==2.1.0 -q
!pip install librosa soundfile pydub -q
!pip install faster-whisper whisper-timestamped -q
!pip install inflect==7.0.0 unidic-lite jieba unidecode eng_to_ipa pypinyin cn2an wavmark -q

if not os.path.isdir('/content/OpenVoice'):
    !git clone https://github.com/myshell-ai/OpenVoice.git

os.makedirs('/content/OpenVoice/checkpoints_v2/converter', exist_ok=True)
!wget -q -O /content/OpenVoice/checkpoints_v2/converter/checkpoint.pth \
    'https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/checkpoint.pth'
!wget -q -O /content/OpenVoice/checkpoints_v2/converter/config.json \
    'https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/main/converter/config.json'

if '/content/OpenVoice' not in sys.path:
    sys.path.insert(0, '/content/OpenVoice')
print('openvoice installed.')

# ---- C3: koko-vc ----
!pip install gradio_client -q
print('VC engines ready.')
print('\n>>> RESTART RUNTIME NOW, then run PHASE D. <<<')


"""## 🟡 PHASE D — Replay deps → Restart Runtime بعدها"""

# ---- D1 ----
!apt-get install -y -q ffmpeg
!pip install numpy scipy librosa soundfile pydub pyroomacoustics ffmpeg-python -q
print('Replay deps installed.')
print('\n>>> RESTART RUNTIME ONE MORE TIME, then re-run A1, then run PHASE E. <<<')


"""## 🟢 PHASE E"""

# Install Silero VAD
!pip install silero-vad

# ---- E1: Imports + Fixes ----
import os, sys, math, random, shutil, gc, traceback, importlib, inspect, types, json
from datetime import datetime
import numpy as np, pandas as pd, librosa
import soundfile as sf
import torch
from tqdm.auto import tqdm
import importlib.util
import hashlib

# Fix 1: huggingface_hub patch
import huggingface_hub
if not hasattr(huggingface_hub, 'is_offline_mode'):
    try:
        from huggingface_hub.constants import HF_HUB_OFFLINE
        huggingface_hub.is_offline_mode = lambda: bool(HF_HUB_OFFLINE)
    except ImportError:
        huggingface_hub.is_offline_mode = lambda: False
print('✅ huggingface_hub patched')

# Fix 2: Spoofing/core __init__.py
SPOOFING_DIR = f'{REPO_DIR}/Spoofing'
core_dir = f'{SPOOFING_DIR}/core'
init_file = f'{core_dir}/__init__.py'
if not os.path.isfile(init_file): open(init_file, 'a').close()
print('✅ Spoofing/core/__init__.py ensured')

# Fix 3: Load dispatchers + VAD
from Spoofing.voice_conversion import run_vc_on_file
from Spoofing.text_to_speech   import run_tts_on_file
import Spoofing.replay_attack  as replay_module
from preprocessing import run_silero_vad
print('✅ Repo dispatchers loaded')


# ---- E2: Paths + Engines + Directories ----

# === TEST split paths ===
SPLIT_PATHS = {
    'test': {
        'source'    : f'{PROJECT_ROOT}/processed/manifest/real_clean_splits/final_split/test_spoof_source_clean.csv',
        'target'    : f'{PROJECT_ROOT}/processed/manifest/spoof_targets_splits/test_spoof_targets.csv',
        'transcript': f'{PROJECT_ROOT}/spoofing/transcripts/test_spoof_c_transcript_inventory.csv',
    },
}

print('🔍 Path check:')
for split, paths in SPLIT_PATHS.items():
    for k, p in paths.items():
        ok = '✅' if os.path.exists(p) else '❌'
        print(f'   {ok} {split}.{k}: {p}')

# === Engines per spoof type (3 each) ===
TEST_ENGINES = {
    'vc'     : ['koko_vc',  'seed_vc',     'openvoice_vc'],
    'tts'    : ['koko_tts', 'xttsv2_tts',  'chatterbox_tts'],
    'replay' : ['replay_c1','replay_c2',   'replay_c3'],
}

def engine_type(eng):
    if eng.endswith('_vc'):       return 'vc'
    if eng.endswith('_tts'):      return 'tts'
    if eng.startswith('replay_'): return 'replay'
    raise ValueError(f'Unknown engine: {eng}')

# === Output / Manifest / JSON / Processed-segments base dirs ===
OUT_BASE       = f'{PROJECT_ROOT}/spoofing/test_data'
MANIFEST_BASE  = f'{PROJECT_ROOT}/spoofing/manifest'
JSON_BASE      = f'{PROJECT_ROOT}/spoofing/intermediate_data/json_configs'
PROCESSED_BASE = f'{PROJECT_ROOT}/spoofing/intermediate_data/processed_segments'

for d in [OUT_BASE, MANIFEST_BASE, JSON_BASE, PROCESSED_BASE]:
    os.makedirs(d, exist_ok=True)

# Pre-create audio output directories
for kind, engines in TEST_ENGINES.items():
    for eng in engines:
        os.makedirs(os.path.join(OUT_BASE, f'test_spoof_{kind}_clean', eng), exist_ok=True)

# === Audio constants ===
CROSS_AGE_P    = 0.5    # متوازنة للتست
TARGET_DUR     = 7.0
SEED           = 42
SR             = 16000
TARGET_SEC     = 3.0
MAX_TGT_TRIES  = 20
MAX_SRC_TRIES  = 50
TARGET_SR      = 16000
MERGE_GAP_MS   = 500
MIN_SPEECH_SEC = 3.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

print(f'\n✅ OUT_BASE:       {OUT_BASE}')
print(f'✅ MANIFEST_BASE:  {MANIFEST_BASE}')
print(f'✅ JSON_BASE:      {JSON_BASE}')
print(f'✅ PROCESSED_BASE: {PROCESSED_BASE}')
print(f'✅ CROSS_AGE_P={CROSS_AGE_P}, TARGET_DUR≥{TARGET_DUR}s, TARGET_SEC={TARGET_SEC}s')


# ---- E3: Load test CSVs ----

def load_split_data(split):
    src = pd.read_csv(SPLIT_PATHS[split]['source'], dtype={'speaker_id': str})
    tgt = pd.read_csv(SPLIT_PATHS[split]['target'], dtype={'speaker_id': str})
    tr  = pd.read_csv(SPLIT_PATHS[split]['transcript'], dtype={'segment_id': str})
    lookup = dict(zip(
        tr[tr['has_transcript'] == 1]['segment_id'].astype(str),
        tr[tr['has_transcript'] == 1]['sentence_clean']
    ))
    return src, tgt, lookup

test_src, test_tgt, test_tr_lookup = load_split_data('test')

print(f'📊 TEST: {len(test_src)} source segs, {len(test_tgt)} target files, {len(test_tr_lookup)} transcripts')
print(f'\n🎯 TEST age distribution:\n{test_src["mapped_age_class"].value_counts().to_string()}')

SPLIT_DATA = {
    'test': {'src': test_src, 'tgt': test_tgt, 'tr': test_tr_lookup},
}


# ---- E4: Manifest schema + helpers ----

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

# Per-spoof-type files
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


# ---- E5: Build JSON configs (auto-calculated counts) ----

def build_json_config(kind, engines, n_src, cross_age_p):
    """
    لكل spoof type:
      - count_required per engine = n_src // len(engines)  (يوزع البواقي)
      - cross_age_count = round(count * cross_age_p)  لـ vc/tts
      - replay: ما له cross_age

    لو الـ config موجود مسبقاً: نرجّعه زي ما هو (نحمي count_done).
    """
    json_path = get_json_config_path(kind)

    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
        print(f'   ⏩ Config exists for test/{kind} — keeping as-is')
        return existing, json_path

    def split_evenly(total, n):
        base = total // n
        rem  = total % n
        return [base + (1 if i < rem else 0) for i in range(n)]

    counts = split_evenly(n_src, len(engines))

    config = {}
    for eng, cnt in zip(engines, counts):
        if kind == 'replay':
            config[eng] = {'count_required': cnt, 'count_done': 0}
        else:
            config[eng] = {
                'count_required'  : cnt,
                'cross_age_count' : round(cnt * cross_age_p),
                'count_done'      : 0,
            }

    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f'   ✅ New config created for test/{kind}')
    return config, json_path


print('\n📐 Building JSON configs for TEST...')
n_src = len(test_src)
print(f'  n_src = {n_src}')
for kind, engines in TEST_ENGINES.items():
    cfg, path = build_json_config(kind, engines, n_src, CROSS_AGE_P)
    print(f'  test/{kind}: {path}')
    for eng, v in cfg.items():
        ca = f", cross_age={v['cross_age_count']}" if 'cross_age_count' in v else ''
        print(f'    {eng}: required={v["count_required"]}{ca}, done={v["count_done"]}')

print('\n✅ All JSON configs ready')


# ---- E6: Audio helpers + engine dispatcher ----

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
        ref = tgt_path or src_path
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


# ---- E7: target selection (diverse + reproducible) ----

def find_valid_target_file(tgt_df, speaker_id, rng, min_duration=TARGET_DUR):
    spk_files = tgt_df[
        (tgt_df['speaker_id'] == speaker_id) &
        (tgt_df['vad_status'] == 'success') &
        (tgt_df['speech_duration_sec'] >= min_duration)
    ]
    if len(spk_files) == 0:
        return None
    idx = rng.integers(0, len(spk_files))
    return spk_files.iloc[idx]['file_path']


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

    # Fallback: any speaker with a valid file
    valid_pool = tgt_df[
        (tgt_df['vad_status'] == 'success') &
        (tgt_df['speech_duration_sec'] >= TARGET_DUR)
    ].copy()
    shuffled_idx = rng.permutation(len(valid_pool))
    valid_pool = valid_pool.iloc[shuffled_idx]

    for _, row in valid_pool.iterrows():
        if os.path.exists(str(row.get('file_path', ''))):
            return row, row['file_path']

    raise RuntimeError('Could not find any valid target file')


# ---- E8: ManifestWriter + row builder ----

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
        'target_file_path'     : (np.nan if is_replay else safe_val(tgt_row.get('file_path') if tgt_row is not None else np.nan)),
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


# ---- E9: MAIN — run_one_test_engine ----

def run_one_test_engine(engine):
    """
    تشغيل engine واحد على التست لحد ما يوصل count_required.

    نفس فلسفة train_valid v3:
    - Shuffle قبل الفلترة (resume بنفس الترتيب)
    - rng deterministic per engine (seeded من SEED + hash اسم الـ engine)
    - target diverse-but-reproducible
    - cross_age flags pre-assigned ومخلوطة
    - VAD على ناتج TTS فقط
    - update JSON + processed + manifest فقط على النجاح
    """
    kind = engine_type(engine)
    src_df = SPLIT_DATA['test']['src'].copy()
    tgt_df = SPLIT_DATA['test']['tgt'].copy()
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

    if count_done >= count_required:
        print(f'🎉 test/{kind}/{engine}: already complete ({count_done}/{count_required})')
        return

    remaining = count_required - count_done
    processed_set = load_processed_set(kind)

    manifest_path = get_manifest_path(kind)
    out_dir       = get_out_dir(kind, engine)
    os.makedirs(out_dir, exist_ok=True)
    create_empty_manifest(manifest_path)

    print(f'\n========== test/{kind}/{engine} | need={remaining}/{count_required} ==========')

    # Shuffle FULL df first (fixed seed), then filter processed
    all_src_shuffled = src_df.sample(frac=1, random_state=SEED)
    available_src    = all_src_shuffled[
        ~all_src_shuffled['segment_id'].astype(str).isin(processed_set)
    ].reset_index(drop=True)

    if kind == 'tts':
        valid_ids = set(tr_lookup.keys())
        available_src = available_src[
            available_src['segment_id'].astype(str).isin(valid_ids)
        ].reset_index(drop=True)

    # Pre-assign cross_age flags
    if kind != 'replay':
        rng_ca = np.random.default_rng(SEED)
        cross_age_flags = np.array(
            [True] * cross_age_count + [False] * (count_required - cross_age_count)
        )
        rng_ca.shuffle(cross_age_flags)
        print(f'   🎲 cross_age_flags: {cross_age_count} True / {count_required - cross_age_count} False')

    # Per-engine rng (deterministic + diverse)
    seed_str = f'test_{kind}_{engine}'
    engine_seed = (
        SEED +
        int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
    )
    rng_target = np.random.default_rng(engine_seed)

    src_tried     = 0
    src_idx       = 0
    ok_count      = 0
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

            # cross_age
            if kind == 'replay':
                cross_age = False
            else:
                flag_idx  = count_done + ok_count
                cross_age = bool(cross_age_flags[flag_idx]) if flag_idx < len(cross_age_flags) else False

            # target
            tgt_row, tgt_path = None, None
            if kind != 'replay':
                try:
                    tgt_row, tgt_path = pick_target(tgt_df, src_age, cross_age, rng=rng_target)
                except RuntimeError as e:
                    print(f'  ⚠️ [{engine}] {seg_id}: no valid target — {e}')
                    continue

            # transcript
            text = None
            if kind == 'tts':
                text = tr_lookup.get(seg_id)
                if not text or (isinstance(text, float) and math.isnan(text)):
                    continue

            # run engine
            out_filename = make_filename(seg_id, engine, kind)
            out_path     = os.path.join(out_dir, out_filename)

            try:
                raw_audio = run_engine(engine, src_path, tgt_path, text)
            except Exception as e:
                print(f'  FAIL [{engine}] src={seg_id} → {type(e).__name__}: {e}')
                continue

            if raw_audio is None:
                continue

            # post-process
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

            # save
            try:
                save_audio(out_path, final_audio, sr=SR, target_sec=TARGET_SEC)
            except Exception as e:
                print(f'  SAVE FAIL [{engine}] {seg_id} → {e}')
                continue

            # success update
            ok_count      += 1
            processed_set.add(seg_id)
            batch_counter += 1
            print(f'✅ [{engine}] saved: {seg_id}')

            config[engine]['count_done'] = count_done + ok_count

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

    final_done = count_done + ok_count
    print(f'✅ test/{kind}/{engine}: done={ok_count}, total={final_done}/{count_required}, src_tried={src_tried}')

    if final_done < count_required:
        print('  ⚠️ Source pool exhausted before reaching target count.')

print('✅ run_one_test_engine ready')


"""## 🚀 RUN — TEST data generation"""

### VC
run_one_test_engine('koko_vc')
run_one_test_engine('seed_vc')
run_one_test_engine('openvoice_vc')

### TTS
run_one_test_engine('koko_tts')
run_one_test_engine('xttsv2_tts')
run_one_test_engine('chatterbox_tts')

### REPLAY
run_one_test_engine('replay_c1')
run_one_test_engine('replay_c2')
run_one_test_engine('replay_c3')


"""## 📊 Status Check"""

print('\n' + '='*70)
print('TEST STATUS CHECK')
print('='*70)
for kind in ['vc', 'tts', 'replay']:
    try:
        cfg = load_json_config(kind)
        print(f'\n[test/{kind}]')
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
        print(f'  ❌ Config not found for test/{kind}')
