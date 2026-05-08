# -*- coding: utf-8 -*-
"""
F5-TTS Local Wrapper
الاستخدام في Colab:
    !pip install -q f5-tts pydub
    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO
    %cd YOUR_REPO
    from f5_tts import run_f5_tts
"""

import uuid
from typing import Optional

_MODEL = None


def get_model(model_type: str = "F5TTS_v1_Base"):
    """تحميل النموذج مرة واحدة فقط (Singleton)"""
    global _MODEL
    if _MODEL is None:
        from core.f5_cloner import F5Clone
        _MODEL = F5Clone(model_type=model_type)
    return _MODEL


def run_f5_tts(
    text: str,
    reference_audio_path: str,
    output_path: Optional[str] = None,
    ref_text: str = "",
    speed: float = 1.0,
    remove_silence: bool = True,
    model_type: str = "F5TTS_v1_Base",
) -> str:
    """
    توليد كلام مستنسخ من صوت مرجعي

    Args:
        text:                 النص المراد تحويله إلى كلام
        reference_audio_path: مسار ملف الصوت المرجعي (WAV / MP3 / M4A)
        output_path:          مسار الحفظ (اختياري)
        ref_text:             نص الصوت المرجعي إن عرفته (اختياري)
        speed:                سرعة الكلام (افتراضي 1.0)
        remove_silence:       إزالة الصمت (افتراضي True)
        model_type:           "F5TTS_v1_Base" | "F5TTS_Base" | "E2TTS_Base"

    Returns:
        str: مسار ملف الصوت الناتج
    """
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
