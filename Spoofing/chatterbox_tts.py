# -*- coding: utf-8 -*-

import uuid
import torch
import torchaudio

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from chatterbox.tts import ChatterboxTTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            _MODEL = ChatterboxTTS.from_pretrained(device=device)
        except Exception:
            _MODEL = ChatterboxTTS.from_pretrained(device="cpu")
    return _MODEL


def _split_into_chunks(text, max_words=50):
    sentences = text.strip().replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        if len((current + " " + sentence).split()) > max_words and current:
            chunks.append(current.strip() + ".")
            current = sentence
        else:
            current = current + ". " + sentence if current else sentence

    if current:
        chunks.append(current.strip() + ".")

    return chunks


def run_chatterbox_tts(
    text: str,
    reference_audio_path: str,
    exaggeration: float = 0.6,
    cfg_weight: float = 0.4,
    max_chunk_words: int = 50,
) -> str:

    model = get_model()

    chunks = _split_into_chunks(text, max_chunk_words)
    wav_tensors = []

    for chunk in chunks:
        wav = model.generate(
            text=chunk,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=reference_audio_path
        )
        wav_tensors.append(wav)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_audio = torch.cat(wav_tensors, dim=1)

    out_path = f"/content/{uuid.uuid4()}_chatterbox_tts.wav"
    torchaudio.save(out_path, full_audio.cpu(), model.sr)

    return out_path
