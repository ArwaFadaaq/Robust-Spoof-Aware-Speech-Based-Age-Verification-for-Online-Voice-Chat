# -*- coding: utf-8 -*-
"""
F5-TTS Voice Cloning Function
المسار في الريبو: Spoofing/f5_tts_function.py
"""

def run_f5_tts(
    ref_audio_path: str,
    gen_text: str,
    ref_text: str = "",
    remove_silence: bool = True,
    output_path: str = "output.wav"
) -> str:

    from gradio_client import Client, handle_file
    from pydub import AudioSegment
    import os, shutil

    ref_wav = "ref_temp.wav"
    if not ref_audio_path.endswith(".wav"):
        AudioSegment.from_file(ref_audio_path)\
            .set_frame_rate(24000)\
            .set_channels(1)\
            .export(ref_wav, format="wav")
    else:
        shutil.copy(ref_audio_path, ref_wav)

    client = Client("mrfakename/E2-F5-TTS")

    result = client.predict(
        ref_audio=handle_file(ref_wav),
        ref_text=ref_text,
        gen_text=gen_text,
        remove_silence=remove_silence,
        api_name="/predict"
    )

    shutil.copy(result, output_path)

    if os.path.exists(ref_wav):
        os.remove(ref_wav)

    return output_path
