
# Chatterbox inference pipeline. Model and code sourced from:
# https://github.com/resemble-ai/chatterbox
# -*- coding: utf-8 -*-
"""

This module performs Text-to-Speech (TTS) voice cloning using ChatterboxTTS.

It:
- Loads the Chatterbox model once and caches it globally.
- Takes input text and a reference speaker audio.
- Splits long text into smaller chunks for stable generation.
- Generates speech conditioned on the reference speaker voice.
- Merges all generated chunks into a single waveform.
- Returns the generated audio as a float32 NumPy waveform.

Temporary WAV files are used internally for resampling and waveform loading.
"""

import os
import uuid
import torch
import torchaudio
import librosa
import numpy as np

# ─────────────────────────────────────────
# MODEL LOADER (singleton)
# ─────────────────────────────────────────

# Global cache variable to avoid reloading the model every call
_MODEL = None


def get_model():
    """
    Load ChatterboxTTS model once and cache it globally.

    Returns:
        Loaded ChatterboxTTS model.
    """
    global _MODEL

    if _MODEL is None:
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ChatterboxTTS] Loading model on device: {device}")

        try:
            _MODEL = ChatterboxTTS.from_pretrained(device=device)
            print("[ChatterboxTTS] Model loaded successfully")

        except Exception:
            print("[ChatterboxTTS] GPU loading failed. Trying CPU fallback...")
            _MODEL = ChatterboxTTS.from_pretrained(device="cpu")
            print("[ChatterboxTTS]  Model loaded on CPU")

    return _MODEL


# ─────────────────────────────────────────
# TEXT CHUNKER
# ─────────────────────────────────────────
def split_into_chunks(text: str, max_words: int = 50) -> list[str]:
    """
    Split long text into smaller sentence-based chunks.

    This helps:
    - Reduce memory usage.
    - Improve generation stability.
    - Avoid failures on very long text.

    Args:
        text      : input text
        max_words : maximum words per chunk

    Returns:
        List of text chunks.
    """

    sentences = text.strip().replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:

        sentence_words = sentence.split()

        if current_word_count + len(sentence_words) > max_words and current_chunk:

            chunks.append(current_chunk.strip() + ".")
            current_chunk = sentence
            current_word_count = len(sentence_words)

        else:

            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence

            current_word_count += len(sentence_words)

    if current_chunk:
        chunks.append(current_chunk.strip() + ".")

    return chunks


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────
def run_chatterbox_tts(
    text: str,
    reference_audio_path: str,
    output_path: str = "output.wav",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    max_chunk_words: int = 50,
    sr: int = 16000,
) -> np.ndarray:
    """
    Generate speech using ChatterboxTTS voice cloning.

    Args:
        text                : input text
        reference_audio_path: reference speaker audio path
        output_path         : unused (kept for compatibility)
        exaggeration        : speaking style intensity
        cfg_weight          : similarity strength to reference speaker
        max_chunk_words     : maximum words per chunk
        sr                  : target output sample rate

    Returns:
        np.ndarray waveform at target sample rate
    """

    print("\n[ChatterboxTTS] Starting generation...")

    # ── Reference audio validation ──────────────────
    if not os.path.exists(reference_audio_path):
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio_path}"
        )

    # Load reference audio only for duration validation
    wav_check, sr_check = torchaudio.load(reference_audio_path)
    duration = wav_check.shape[1] / sr_check
    model = get_model()
    chunks = split_into_chunks(text, max_chunk_words)


    # ── Generation ──────────────────
    wav_tensors = []

    for i, chunk in enumerate(chunks):

        try:
            wav = model.generate(
                text=chunk,
                audio_prompt_path=reference_audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

            wav_tensors.append(wav)

        except Exception as e:
            print(f"[ChatterboxTTS] Chunk {i+1} failed: {e}")
            continue

        # Clear GPU cache between chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not wav_tensors:
        raise RuntimeError("No audio generated")

    full_audio = torch.cat(wav_tensors, dim=1)
    tmp_path = f"/tmp/{uuid.uuid4()}_cb.wav"

    torchaudio.save(tmp_path, full_audio, model.sr)

    audio, _ = librosa.load(tmp_path, sr=sr)

    print("[ChatterboxTTS] Generation completed successfully")

    return audio.astype(np.float32)
