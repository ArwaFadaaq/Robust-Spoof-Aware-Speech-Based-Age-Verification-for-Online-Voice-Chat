# -*- coding: utf-8 -*-

import os
import uuid
import shutil

_CLIENT = None


def get_model():
    global _CLIENT
    if _CLIENT is None:
        from gradio_client import Client
        _CLIENT = Client("mrfakename/E2-F5-TTS")
    return _CLIENT


def run_f5_tts(
    ref_audio_path: str,
    gen_text: str,
    ref_text: str = "",
    remove_silence: bool = True,
) -> str:

    from gradio_client import handle_file

    client = get_model()

    ref_wav = f"/content/{uuid.uuid4()}_ref.wav"
    out_path = f"/content/{uuid.uuid4()}_f5_tts.wav"

    shutil.copy(ref_audio_path, ref_wav)

    result = client.predict(
        ref_audio=handle_file(ref_wav),
        ref_text=ref_text,
        gen_text=gen_text,
        remove_silence=remove_silence,
        api_name="/predict"
    )

    shutil.copy(result, out_path)

    if os.path.exists(ref_wav):
        os.remove(ref_wav)

    return out_path
 
