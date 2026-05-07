# -*- coding: utf-8 -*-
"""
F5-TTS Voice Cloner Core
يشتغل محلياً على GPU الـ Colab بدون أي quota خارجي
"""

import os
import torch
import torchaudio
import tempfile
from pathlib import Path
from pydub import AudioSegment


class F5Clone:
    """
    Wrapper نظيف لنموذج F5-TTS للاستخدام المحلي في Colab
    """

    SUPPORTED_LANGS = ["en", "zh", "ar"]  # English, Chinese, Arabic (partial)

    def __init__(self, model_type: str = "F5TTS_v1_Base", device: str = None):
        """
        تهيئة النموذج

        Args:
            model_type: نوع النموذج - "F5TTS_v1_Base" أو "F5TTS_Base" أو "E2TTS_Base"
            device: "cuda" أو "cpu" - يتحدد تلقائياً لو تركته فارغ
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._vocoder = None
        print(f"[F5Clone] Device: {self.device} | Model: {self.model_type}")

    def _load(self):
        """تحميل النموذج عند أول استخدام (lazy loading)"""
        if self._model is not None:
            return

        try:
            from f5_tts.model import DiT, UNetT
            from f5_tts.infer.utils_infer import (
                load_model,
                load_vocoder,
                preprocess_ref_audio_text,
                infer_process,
            )
        except ImportError:
            raise ImportError(
                "f5-tts غير مثبت. شغّل: !pip install -q f5-tts"
            )

        print(f"[F5Clone] جاري تحميل النموذج {self.model_type} ...")

        # إعداد معمارية النموذج
        if "E2TTS" in self.model_type:
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            repo_id = "SWivid/F5-TTS"
            ckpt_file = "E2TTS_Base/model_1200000.safetensors"
        elif self.model_type == "F5TTS_v1_Base":
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            repo_id = "SWivid/F5-TTS"
            ckpt_file = "F5TTS_v1_Base/model_1250000.safetensors"
        else:  # F5TTS_Base
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            repo_id = "SWivid/F5-TTS"
            ckpt_file = "F5TTS_Base/model_1200000.safetensors"

        self._model = load_model(model_cls, model_cfg, ckpt_file, repo_id=repo_id)
        self._model = self._model.to(self.device)
        self._vocoder = load_vocoder(is_local=False)

        # حفظ الدوال المساعدة
        self._preprocess = preprocess_ref_audio_text
        self._infer = infer_process

        print("[F5Clone] ✅ النموذج جاهز!")

    def _to_wav(self, audio_path: str) -> str:
        """تحويل أي صيغة صوتية إلى WAV 24kHz mono"""
        path = Path(audio_path)
        if path.suffix.lower() == ".wav":
            return audio_path
        out = tempfile.mktemp(suffix=".wav")
        AudioSegment.from_file(audio_path).set_frame_rate(24000).set_channels(1).export(out, format="wav")
        return out

    def generate(
        self,
        text: str,
        reference_audio: str,
        output_path: str = None,
        ref_text: str = "",
        speed: float = 1.0,
        remove_silence: bool = True,
        cross_fade_duration: float = 0.15,
        nfe_step: int = 32,
    ) -> str:
        """
        توليد صوت مستنسخ

        Args:
            text:             النص المراد تحويله
            reference_audio:  مسار ملف الصوت المرجعي (WAV/MP3/M4A ...)
            output_path:      مسار الملف الناتج (اختياري - يُنشأ تلقائياً)
            ref_text:         نص الصوت المرجعي (اختياري - يُستخرج تلقائياً)
            speed:            سرعة الكلام (0.5 - 2.0)
            remove_silence:   إزالة الصمت التلقائية
            cross_fade_duration: مدة التلاشي بين المقاطع (ثانية)
            nfe_step:         خطوات الـ inference (أقل = أسرع، أكثر = أجود)

        Returns:
            str: مسار ملف الصوت الناتج
        """
        self._load()

        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        # تحضير الصوت المرجعي
        ref_wav = self._to_wav(reference_audio)

        # تحضير المدخلات
        ref_audio_tensor, ref_text_out = preprocess_ref_audio_text(
            ref_wav, ref_text, show_info=print
        )

        # توليد الصوت
        audio_out, sample_rate, _ = infer_process(
            ref_audio=ref_audio_tensor,
            ref_text=ref_text_out,
            gen_text=text,
            model_obj=self._model,
            vocoder=self._vocoder,
            cross_fade_duration=cross_fade_duration,
            speed=speed,
            show_info=print,
            progress=None,
            nfe_step=nfe_step,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            device=self.device,
        )

        # حفظ الناتج
        if output_path is None:
            import uuid
            output_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torchaudio.save(output_path, audio_out.unsqueeze(0).cpu(), sample_rate)

        print(f"[F5Clone] ✅ تم الحفظ: {output_path}")
        return output_path
