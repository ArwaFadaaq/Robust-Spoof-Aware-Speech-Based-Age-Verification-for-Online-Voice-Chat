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


def run_XTTSv2_tts(text: str, reference_audio_path: str) -> str:
    model = _load_model()

    out_path = f"/content/{uuid.uuid4()}_xtts.wav"

    model.tts_to_file(
        text=text,
        speaker_wav=reference_audio_path,
        language="en",
        file_path=out_path
    )

    return out_path
