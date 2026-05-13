import os, torch
import numpy as np

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_model():
    global _MODEL
    if _MODEL is None:
        from transformers import AutoProcessor, BarkModel
        _MODEL = {
            "processor": AutoProcessor.from_pretrained("suno/bark-small"),
            "model": BarkModel.from_pretrained("suno/bark-small").to(_DEVICE)
        }
    return _MODEL

def run_XTTSv2_tts(text: str, target_wav: str, out_path: str) -> np.ndarray:
    m = _load_model()
    inputs = m["processor"](text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        audio = m["model"].generate(**inputs)
    audio_np = audio.cpu().numpy().squeeze().astype(np.float32)
    if audio_np.max() > 1.0:
        audio_np = audio_np / 32768.0
    return audio_np

run_tts_on_file = run_XTTSv2_tts
