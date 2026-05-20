# -*- coding: utf-8 -*-
"""
Voice Conversion (VC) Engine Dispatcher

This module provides a unified interface for multiple Voice Conversion (VC)
models.

It:
- Loads source and target audio files.
- Routes them to the selected VC engine.
- Supports multiple backends (Koko, Seed-VC, OpenVoice).
- Returns a converted waveform as a NumPy array.

This design allows swapping VC models without modifying the main pipeline.
"""

import librosa

# Import available voice conversion engines
from Spoofing.koko_vc import run_koko
from Spoofing.seed_vc import run_seed
from Spoofing.openvoice_vc import run_openvoice


# ─────────────────────────────────────────
# ENGINE REGISTRY
# ─────────────────────────────────────────

# Mapping between model name and VC function
VC_ENGINES = {
    "koko": run_koko,
    "seed": run_seed,
    "openvoice": run_openvoice,
}


# ─────────────────────────────────────────
# MAIN DISPATCH FUNCTION
# ─────────────────────────────────────────
def run_vc_on_file(source_path, target_path, model_name, sr=16000):
    """
    Run voice conversion using the selected engine.

    Args:
        source_path : path to source speaker audio
        target_path : path to target speaker audio
        model_name  : VC model to use (koko, seed, openvoice)
        sr          : sample rate for loading audio

    Returns:
        np.ndarray converted audio waveform
    """

    # Load source and target audio
    src_audio, _ = librosa.load(source_path, sr=sr)
    tgt_audio, _ = librosa.load(target_path, sr=sr)

    # Route to selected VC engine
    return VC_ENGINES[model_name](src_audio, tgt_audio, sr)
