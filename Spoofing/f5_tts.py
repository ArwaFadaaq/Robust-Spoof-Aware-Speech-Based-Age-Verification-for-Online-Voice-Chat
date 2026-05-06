def run_f5_tts(
    text: str,
    reference_audio_path: str,
    ref_text: str = "",
    remove_silence: bool = True,
) -> str:

    from gradio_client import handle_file

    client = get_model()

    ref_wav = f"/content/{uuid.uuid4()}_ref.wav"
    out_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    shutil.copy(reference_audio_path, ref_wav)

    result = client.predict(
        ref_audio=handle_file(ref_wav),
        ref_text=ref_text,
        gen_text=text,
        remove_silence=remove_silence,
        api_name="/predict"
    )

    shutil.copy(result, out_path)

    if os.path.exists(ref_wav):
        os.remove(ref_wav)

    # ✅ تأكيد أن الصوت موجود وصالح
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("F5 TTS failed: empty output audio")

    return out_path
