# -*- coding: utf-8 -*-

import numpy as np

from koko_tts import run_koko_tts
from chatterbox_tts import run_chatterbox_tts
from XTTSv2_tts import run_XTTSv2_tts

TTS_ENGINES = {
    "koko": run_koko_tts,
    "chatterbox": run_chatterbox_tts,
    "xttsv2": run_XTTSv2_tts
}


def run_tts_on_file(
    text: str,
    reference_audio_path: str,
    model_name: str,
) -> np.ndarray:

    return TTS_ENGINES[model_name](
        text=text,
        reference_audio_path=reference_audio_path,
    )
