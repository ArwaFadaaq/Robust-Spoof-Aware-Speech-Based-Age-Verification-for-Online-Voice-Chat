def run_koko_tts(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
) -> str:

    model = get_model()

    out_path = f"/content/{uuid.uuid4()}_koko_tts.wav"

    model.generate(
        text=text,
        lang=lang,
        reference_audio=reference_audio_path,
        output_path=out_path
    )

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("Koko TTS failed: empty or missing output audio")

    return out_path
