# -*- coding: utf-8 -*-
"""
seed_vc.ipynb

This module performs Voice Conversion (VC) using Seed-VC.

It:
- Loads the Seed-VC inference pipeline once and caches it globally.
- Takes a source speaker audio and a target speaker audio.
- Converts the source speech to match the target speaker characteristics.
- Uses the repository default checkpoints and inference settings.
- Returns the generated audio as a float32 waveform (NumPy array).

The function internally saves temporary WAV files because the Seed-VC
inference pipeline operates on file paths.
"""

import os
import sys
import uuid
import soundfile as sf
import librosa
import numpy as np


SEED_DIR = "/content/seed-vc"
if SEED_DIR not in sys.path:
    sys.path.append(SEED_DIR)


# =========================
# GLOBAL CACHE
# =========================

# Global variable to hold the loaded inference function and arguments.
# This prevents reloading the model configuration on every call.
_MODEL = None


# Loads the Seed-VC inference function and default arguments once,
# then stores them in memory for reuse.
def _load_model_once():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    from inference_v2 import convert_voice_v2

    print("[SeedVC] Loading model first time...")

    class Args:
        def __init__(self):
            self.diffusion_steps = 50
            self.similarity_cfg_rate = 0.75
            self.intelligibility_cfg_rate = 0.7
            self.length_adjust = 1.0
            self.top_p = 0.9
            self.temperature = 1.0
            self.repetition_penalty = 1.0
            self.convert_style = True
            self.anonymization_only = False

            self.compile = False

            # IMPORTANT: use repo default loaders (NO manual checkpoint path)
            self.ar_checkpoint_path = None
            self.cfm_checkpoint_path = None

    _MODEL = {
        "fn": convert_voice_v2,
        "args": Args()
    }

    print("[SeedVC] Model loaded")

    return _MODEL


# =========================
# MAIN FUNCTION
# =========================

# Takes a source audio and a target audio, performs voice conversion
# using Seed-VC, and returns the converted audio as float32.
def run_seed(src_audio, tgt_audio, sr=16000):

    print("\n[SeedVC] Running conversion...")

    model = _load_model_once()
    fn = model["fn"]
    args = model["args"]

    # Generate unique temp file paths to avoid collisions
    uid = str(uuid.uuid4())[:8]

    src_path = f"/tmp/src_{uid}.wav"
    tgt_path = f"/tmp/tgt_{uid}.wav"
    out_path = f"/tmp/out_{uid}.wav"

    # Save input waveforms as temporary WAV files
    sf.write(src_path, src_audio, sr)
    sf.write(tgt_path, tgt_audio, sr)

    # Change working directory because Seed-VC expects repo-relative paths
    old_cwd = os.getcwd()
    os.chdir(SEED_DIR)

    try:
        # Run Seed-VC inference
        result = fn(src_path, tgt_path, args)

        if result is None:
            raise RuntimeError("Seed-VC returned None")

        # Extract generated waveform and sample rate
        sr_out, audio = result

        # Save generated output temporarily
        sf.write(out_path, audio, sr_out)

        print("[SeedVC] Conversion completed successfully")

        # Reload audio at target sample rate and return as float32
        audio, _ = librosa.load(out_path, sr=sr)
        return audio.astype(np.float32)

    finally:
        # Restore original working directory
        os.chdir(old_cwd)

        # Remove temporary files
        if os.path.exists(src_path):
            os.remove(src_path)

        if os.path.exists(tgt_path):
            os.remove(tgt_path)

        if os.path.exists(out_path):
            os.remove(out_path)
