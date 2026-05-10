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
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "f5-tts غير مثبت. شغّل: !pip install -q f5-tts"
            )

        print(f"[F5Clone] جاري تحميل النموذج {self.model_type} ...")

        repo_id = "SWivid/F5-TTS"

        if "E2TTS" in self.model_type:
            model_cls = UNetT
            model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
            ckpt_file = "E2TTS_Base/model_1200000.safetensors"
        elif self.model_type == "F5TTS_v1_Base":
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            ckpt_file = "F5TTS_v1_Base/model_1250000.safetensors"
        else:
            model_cls = DiT
            model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
            ckpt_file = "F5TTS_Base/model_1200000.safetensors"

        ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_file)
        self._model = load_model(model_cls, model_cfg, ckpt_path)
        self._model = self._model.to(self.device)
        self._vocoder = load_vocoder(is_local=False)

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

    def _normalize_ref_duration(self, audio_path: str) -> str:
        """
        يطبّع مدة صوت المرجع لتكون دايماً بين 5-8 ثواني.

        الهدف: النموذج يشوف نفس "كمية" الصوت بغض النظر عن
        طول المقطع اللي أرسلته (3 ث، 6 ث، 12 ث، ...).
        هذا يمنع النموذج من تفسير الطول كمؤشر للسرعة.

        - أقل من 5 ث  → يمدّه للـ 5 (بدون تغيير الـ pitch)
        - بين 5-8 ث   → يبقى كما هو ✅
        - أكثر من 8 ث → يضغطه للـ 8 (بدون تغيير الـ pitch)
        """
        audio = AudioSegment.from_file(audio_path)
        current_duration = len(audio) / 1000  # تحويل من ms إلى ثواني

        target_duration = max(5.0, min(current_duration, 8.0))

        # لو الصوت أصلاً في النطاق المطلوب، ما نحتاج نغير شيء
        if abs(current_duration - target_duration) < 0.1:
            return audio_path

        # تغيير السرعة عن طريق تغيير الـ frame_rate مؤقتاً
        # هذا يحافظ على الـ pitch ويغير السرعة فقط
        ratio = current_duration / target_duration
        normalized = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * ratio)}
        ).set_frame_rate(audio.frame_rate)

        out = tempfile.mktemp(suffix=".wav")
        normalized.export(out, format="wav")

        print(f"[F5Clone] 🎙️ مدة المرجع: {current_duration:.1f}s → {target_duration:.1f}s")
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

        self._load()

        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        # 1) تحويل الصيغة إلى WAV
        ref_wav = self._to_wav(reference_audio)

        # 2) تطبيع مدة المرجع (الإضافة الجديدة)
        ref_wav = self._normalize_ref_duration(ref_wav)

        ref_audio_tensor, ref_text_out = preprocess_ref_audio_text(
            ref_wav, ref_text, show_info=print
        )

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
            sway_sampling_coef=0.0,   # تغيير: كان -1.0 (عشوائي) → 0.0 (ثابت)
            device=self.device,
        )

        if output_path is None:
            import uuid
            output_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        import numpy as np

        if isinstance(audio_out, np.ndarray):
            audio_out = torch.from_numpy(audio_out)

        if audio_out.dim() == 1:
            audio_out = audio_out.unsqueeze(0)

        torchaudio.save(
            output_path,
            audio_out.float().cpu(),
            sample_rate
        )

        print(f"[F5Clone] : {output_path}")
        return output_path
