# -*- coding: utf-8 -*-

from koko_tts import run_koko_tts
from f5_tts_local import run_f5_tts
from chatterbox_tts import run_chatterbox_tts
from coqui_tts import run_coqui_tts

TTS_ENGINES = {
    "koko": run_koko_tts,
    "f5": run_f5_tts,
    "chatterbox": run_chatterbox_tts,
    "coqui": run_coqui_tts
}

def run_tts_on_file(
    text: str,
    reference_audio_path: str,
    model_name: str,
) -> str:

    return TTS_ENGINES[model_name](
        text=text,
        reference_audio_path=reference_audio_path,
    )
