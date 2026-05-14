import uuid
import librosa
import numpy as np
import torch

_MODEL = None

def _load_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        ).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return _MODEL


def run_XTTSv2_tts(
    text: str,
    reference_audio_path: str,
    sr: int = 16000
) -> np.ndarray:

    model = _load_model()

    out_path = f"/content/{uuid.uuid4()}_xtts.wav"

    model.tts_to_file(
        text=text,
        speaker_wav=reference_audio_path,
        language="en",
        file_path=out_path
    )

    audio, _ = librosa.load(out_path, sr=sr)

    return audio.astype(np.float32)
