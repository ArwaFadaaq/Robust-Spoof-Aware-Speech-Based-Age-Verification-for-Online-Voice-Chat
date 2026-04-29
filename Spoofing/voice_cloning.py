# -*- coding: utf-8 -*-

from Spoofing.koko_tts import run_koko_tts
from Spoofing.f5_tts import run_f5_tts
from Spoofing.chatterbox_tts import run_chatterbox_tts


TTS_ENGINES = {
    "koko": run_koko_tts,
    "f5": run_f5_tts,
    "chatterbox": run_chatterbox_tts
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
