import os
import uuid
import torch
from TTS.api import TTS

_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _MODEL

def run_XTTSv2_tts(text: str, target_wav: str, out_path: str):
    model = _load_model()
    model.tts_to_file(
        text=text,
        speaker_wav=target_wav,
        language="en",
        file_path=out_path
    )

run_tts_on_file = run_XTTSv2_tts
