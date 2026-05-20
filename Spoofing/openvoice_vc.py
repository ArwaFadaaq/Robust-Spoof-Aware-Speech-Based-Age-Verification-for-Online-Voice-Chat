# -*- coding: utf-8 -*-
"""
openvoice_vc.py
Spoofing/openvoice_vc.py

This module performs Voice Conversion (VC) using OpenVoice.

It:
- Loads the OpenVoice ToneColorConverter model once and caches it.
- Takes a source speaker audio and a target speaker audio.
- Extracts speaker embeddings (style/tone characteristics) from both.
- Converts the source speech to sound like the target speaker while
  preserving the original linguistic content.
- Returns the generated audio as a float32 waveform (NumPy array).

The function uses temporary WAV files internally because the OpenVoice
API operates on file paths rather than raw arrays.
"""

import os
import sys
import uuid
import librosa
import numpy as np
import soundfile as sf

# Path to the local OpenVoice repository
OPENVOICE_DIR = "/content/OpenVoice"
if OPENVOICE_DIR not in sys.path:
    sys.path.insert(0, OPENVOICE_DIR)

# =========================
# GLOBAL CACHE
# =========================

# Global variable to hold the loaded model; avoids reloading on every call
_MODEL = None

# Loads the ToneColorConverter model and se_extractor once and caches them globally.
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
    print("[OpenVoice] Model loaded")
    return _MODEL

# =========================
# MAIN FUNCTION
# =========================

# Takes a source audio and a target audio, converts the tone color of the source
# to match the target speaker using OpenVoice, and returns the converted audio as float32.
def run_openvoice(src_audio, tgt_audio, sr=16000):
    print("\n[OpenVoice] Running conversion...")
    model = _load_model_once()
    converter   = model["converter"]
    se_extractor = model["se_extractor"]

    # Generate unique temp file paths to avoid conflicts between concurrent calls
    uid      = str(uuid.uuid4())[:8]
    src_path = f"/tmp/src_{uid}.wav"
    tgt_path = f"/tmp/tgt_{uid}.wav"
    out_path = f"/tmp/out_{uid}.wav"

    # Write input arrays to disk as wav files for the OpenVoice API
    sf.write(src_path, src_audio, sr)
    sf.write(tgt_path, tgt_audio, sr)

    # Change working directory to OpenVoice root; required for internal relative imports
    old_cwd = os.getcwd()
    os.chdir(OPENVOICE_DIR)
    try:
        # Extract speaker style embeddings from target and source
        tgt_se, _ = se_extractor.get_se(tgt_path, converter, vad=False)
        src_se, _ = se_extractor.get_se(src_path, converter, vad=False)

        # Run tone color conversion and write result to out_path
        converter.convert(
            audio_src_path=src_path,
            src_se=src_se,
            tgt_se=tgt_se,
            output_path=out_path
        )

        # Load converted audio and return as float32 array
        audio, _ = librosa.load(out_path, sr=sr)
        return audio.astype(np.float32)
    finally:
        # Restore original working directory
        os.chdir(old_cwd)
        # Remove all temporary files
        for p in [src_path, tgt_path, out_path]:
            if os.path.exists(p):
                os.remove(p)
