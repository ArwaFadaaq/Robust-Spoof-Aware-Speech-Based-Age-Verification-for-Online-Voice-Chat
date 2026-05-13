# -*- coding: utf-8 -*-

import uuid
import torch
from TTS.api import TTS

_MODEL = None


def get_model():
    global _MODEL

    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _MODEL = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ).to(device)

    return _MODEL


def run_coqui_tts(
    text: str,
    reference_audio_path: str,
    language: str = "en",
) -> str:

    model = get_model()

    out_path = f"/content/{uuid.uuid4()}_XTTSv2_tts.wav"

    model.tts_to_file(
        text=text,
        speaker_wav=reference_audio_path,
        language=language,
        file_path=out_path,
    )

    return out_path
