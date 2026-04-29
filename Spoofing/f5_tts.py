def run_f5_tts(
    text: str,                    # ← كان gen_text
    reference_audio_path: str,    # ← كان ref_audio_path
    ref_text: str = "",
    remove_silence: bool = True,
) -> str:

    from gradio_client import handle_file
    client = get_model()

    ref_wav = f"/content/{uuid.uuid4()}_ref.wav"
    out_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    shutil.copy(reference_audio_path, ref_wav)   # ← كان ref_audio_path

    result = client.predict(
        ref_audio=handle_file(ref_wav),
        ref_text=ref_text,
        gen_text=text,                           # ← كان gen_text
        remove_silence=remove_silence,
        api_name="/predict"
    )

    shutil.copy(result, out_path)
    if os.path.exists(ref_wav):
        os.remove(ref_wav)

    return out_path
