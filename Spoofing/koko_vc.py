# -*- coding: utf-8 -*-
"""
koko_vc.ipynb

This module performs Voice Conversion (VC) using KokoClone.

It:
- Loads the KokoClone VC model once and caches it globally.
- Takes a source speaker audio and a target speaker audio.
- Converts the source speech to sound similar to the target speaker.
- Saves temporary WAV files required by the KokoClone API.
- Returns the generated audio as a float32 waveform (NumPy array).

"""

import uuid
import soundfile as sf
import librosa
import numpy as np


# =========================
# GLOBAL CACHE
# =========================

# Global variable used to cache the loaded KokoClone model
# so it is not reloaded on every conversion call.
_MODEL = None


# Loads the KokoClone model once and reuses it for future calls.
def get_model():
    global _MODEL

    if _MODEL is None:
        print("[KokoVC] Loading model first time...")

        from core.cloner import KokoClone
        _MODEL = KokoClone()

        print("[KokoVC] Model loaded")

    return _MODEL


# =========================
# MAIN FUNCTION
# =========================

# Takes a source audio and a target audio, performs voice conversion
# using KokoClone, and returns the converted waveform as float32.
def run_koko(src_audio, tgt_audio, sr=16000):

    print("\n[KokoVC] Running conversion...")

    model = get_model()

    # Generate unique temporary file names
    # to avoid collisions between multiple runs.
    src_path = f"/content/{uuid.uuid4()}_src.wav"
    tgt_path = f"/content/{uuid.uuid4()}_tgt.wav"
    out_path = f"/content/{uuid.uuid4()}_out.wav"

    # Save source and target audio arrays as WAV files
    sf.write(src_path, src_audio, sr)
    sf.write(tgt_path, tgt_audio, sr)

    # Run voice conversion
    model.convert(
        source_audio=src_path,
        reference_audio=tgt_path,
        output_path=out_path
    )

    print("[KokoVC] Conversion completed successfully")

    # Load generated audio and resample if needed
    audio, _ = librosa.load(out_path, sr=sr)

    # Return waveform as float32 NumPy array
    return audio.astype(np.float32)
