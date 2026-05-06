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
        if not chunk or len(chunk.strip()) == 0:
            continue

        wav = model.generate(
            text=chunk,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            audio_prompt_path=reference_audio_path
        )

        # ✅ تأكيد أن الناتج tensor صحيح
        if wav is None:
            continue

        wav_tensors.append(wav)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(wav_tensors) == 0:
        raise RuntimeError("Chatterbox TTS failed: no audio generated")

    full_audio = torch.cat(wav_tensors, dim=1)

    if full_audio.numel() == 0:
        raise RuntimeError("Chatterbox TTS produced empty tensor")

    out_path = f"/content/{uuid.uuid4()}_chatterbox_tts.wav"
    torchaudio.save(out_path, full_audio.cpu(), model.sr)

    # ✅ تأكيد الملف
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Chatterbox TTS failed: empty wav file")

    return out_path
