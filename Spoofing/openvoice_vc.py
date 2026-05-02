# -*- coding: utf-8 -*-
"""openvoice_vc.py
Spoofing/openvoice_vc.py
"""
import os
import sys
import uuid
import soundfile as sf

# مسار OpenVoice
OPENVOICE_DIR = "/content/OpenVoice"
if OPENVOICE_DIR not in sys.path:
    sys.path.insert(0, OPENVOICE_DIR)

# =========================
# GLOBAL CACHE
# =========================
_MODEL = None

def _load_model_once():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print("[OpenVoice] Loading model first time...")
    import torch
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

    device = "cuda" if torch.cuda.is_available() else "cpu"

    converter = ToneColorConverter(
        f"{OPENVOICE_DIR}/checkpoints_v2/converter/config.json",
        device=device
    )
    converter.load_ckpt(f"{OPENVOICE_DIR}/checkpoints_v2/converter/checkpoint.pth")

    _MODEL = {
        "converter": converter,
        "se_extractor": se_extractor,
    }

    print("[OpenVoice] ✅ Model loaded")
    return _MODEL

# =========================
# MAIN FUNCTION
# =========================
def run_openvoice(src_audio, tgt_audio, sr=16000):
    print("\n[OpenVoice] Running conversion...")
    model = _load_model_once()

    converter   = model["converter"]
    se_extractor = model["se_extractor"]

    uid      = str(uuid.uuid4())[:8]
    src_path = f"/tmp/src_{uid}.wav"
    tgt_path = f"/tmp/tgt_{uid}.wav"
    out_path = f"/tmp/out_{uid}.wav"

    sf.write(src_path, src_audio, sr)
    sf.write(tgt_path, tgt_audio, sr)

    old_cwd = os.getcwd()
    os.chdir(OPENVOICE_DIR)          # نفس حيلة seed_vc لأن OpenVoice يحتاج CWD صح
    try:
        tgt_se, _ = se_extractor.get_se(tgt_path, converter, vad=False)
        src_se, _ = se_extractor.get_se(src_path, converter, vad=False)

        converter.convert(
            audio_src_path=src_path,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=out_path
        )
        return out_path

    finally:
        os.chdir(old_cwd)
        # cleanup
        for p in [src_path, tgt_path]:
            if os.path.exists(p):
                os.remove(p)
