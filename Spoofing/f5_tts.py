# -*- coding: utf-8 -*-

import os
import uuid
from f5_tts.api import F5TTS

_model = None

def get_model():
    global _model
    if _model is None:
        _model = F5TTS()   # تشغيل محلي
    return _model


def run_f5_tts(
    text: str,
    reference_audio_path: str,
    ref_text: str = "",
    remove_silence: bool = True,
) -> str:

    out_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    model = get_model()

    wav = model.infer(
        ref_audio=reference_audio_path,
        ref_text=ref_text,
        gen_text=text,
        remove_silence=remove_silence,
    )

    # حفظ النتيجة
    import soundfile as sf
    sf.write(out_path, wav, 24000)

    return out_path
