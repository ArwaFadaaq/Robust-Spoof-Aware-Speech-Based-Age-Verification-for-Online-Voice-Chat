# -*- coding: utf-8 -*-

import os
import uuid
import soundfile as sf

_model = None


def get_model():
    global _model
    if _model is None:
        # هذا يعتمد على النسخة المثبتة عندك من F5-TTS
        from f5_tts.infer.utils_infer import load_model
        _model = load_model("F5TTS_v1_Base")
    return _model


def run_f5_tts(
    text: str,
    reference_audio_path: str,
    ref_text: str = "",
    remove_silence: bool = True,
):

    out_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    model = get_model()

    wav, sr = model.infer(
        ref_audio=reference_audio_path,
        ref_text=ref_text,
        gen_text=text,
        remove_silence=remove_silence,
    )

    sf.write(out_path, wav, sr)

    return out_path
