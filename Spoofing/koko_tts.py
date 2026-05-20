# -*- coding: utf-8 -*-
"""
koko_tts.py

This module performs Text-to-Speech (TTS) generation using KokoClone.

It:
- Loads the KokoClone TTS model once and caches it globally.
- Takes input text and a reference speaker audio.
- Generates speech conditioned on the reference speaker voice.
- Saves the generated output temporarily as a WAV file.
- Loads and returns the generated speech as a float32 waveform
  (NumPy array).

The generated speech attempts to preserve the vocal characteristics
of the reference speaker while speaking the provided text.
"""

import uuid
import librosa
import numpy as np


# =========================
# GLOBAL CACHE
# =========================

# Global variable used to cache the loaded KokoClone model
# so it does not reload for every inference call.
_MODEL = None


# Loads the KokoClone model once and reuses it across calls.
def get_model():
    global _MODEL

    if _MODEL is None:
        print("[KokoTTS] Loading model first time...")

        from core.cloner import KokoClone
        _MODEL = KokoClone()

        print("[KokoTTS] Model loaded")

    return _MODEL


# =========================
# MAIN FUNCTION
# =========================

# Takes input text and a reference speaker audio path,
# generates speech using KokoClone TTS,
# and returns the waveform as float32.
def run_koko_tts(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
    sr: int = 16000
) -> np.ndarray:

    print("\n[KokoTTS] Generating speech...")

    model = get_model()

    # Generate a unique temporary output file path
    # to avoid filename collisions.
    out_path = f"/content/{uuid.uuid4()}_koko_tts.wav"

    # Run TTS generation using the reference speaker audio
    model.generate(
        text=text,
        lang=lang,
        reference_audio=reference_audio_path,
        output_path=out_path
    )

    print("[KokoTTS] Speech generation completed successfully")

    # Load generated audio and resample if needed
    audio, _ = librosa.load(out_path, sr=sr)

    # Return waveform as float32 NumPy array
    return audio.astype(np.float32)
