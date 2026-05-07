# -*- coding: utf-8 -*-

import uuid
from typing import Optional

_MODEL = None


def get_model(model_type: str = "F5TTS_v1_Base"):

    global _MODEL

    if _MODEL is None:
        from core.f5_cloner import F5Clone
        _MODEL = F5Clone(model_type=model_type)

    return _MODEL


def run_f5_tts(
    text: str,
    reference_audio_path: str,
    output_path: Optional[str] = None,
    ref_text: Optional[str] = None,
    speed: float = 1.0,
    remove_silence: bool = True,
    model_type: str = "F5TTS_v1_Base",
) -> str:

    # =========================
    # 🚨 منع الفراغ نهائيًا
    # =========================
    if ref_text is None or str(ref_text).strip() == "":
        raise ValueError(
            "❌ Missing ref_text: You must provide transcript from dataset. "
            "Whisper fallback is disabled."
        )

    if output_path is None:
        output_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    model = get_model(model_type=model_type)

    return model.generate(
        text=text,
        reference_audio=reference_audio_path,
        output_path=output_path,
        ref_text=ref_text,
        speed=speed,
        remove_silence=remove_silence,
    )
