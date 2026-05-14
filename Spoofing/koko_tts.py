# -*- coding: utf-8 -*-

import uuid
import librosa
import numpy as np

_MODEL = None


def get_model():
    global _MODEL

    if _MODEL is None:
        from core.cloner import KokoClone
        _MODEL = KokoClone()

    return _MODEL


def run_koko_tts(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
    sr: int = 16000
) -> np.ndarray:

    model = get_model()

    out_path = f"/content/{uuid.uuid4()}_koko_tts.wav"

    model.generate(
        text=text,
        lang=lang,
        reference_audio=reference_audio_path,
        output_path=out_path
    )

    audio, _ = librosa.load(out_path, sr=sr)

    return audio.astype(np.float32)
