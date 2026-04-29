import os
import shutil
import subprocess


def run_koko(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
    output_path: str = "koko_output.wav"
) -> str:

    # convert reference to WAV before anything else
    from pydub import AudioSegment
    ref_wav = "/tmp/reference_converted.wav"
    AudioSegment.from_file(reference_audio_path)\
        .set_frame_rate(24000)\
        .set_channels(1)\
        .export(ref_wav, format="wav")

    if not os.path.exists("kokoclone"):
        subprocess.run(
            ["git", "clone", "https://github.com/Ashish-Patnaik/kokoclone.git"],
            check=True
        )

    os.chdir("kokoclone")

    if not os.path.exists(".deps_installed"):
        subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
        subprocess.run(["pip", "install", "-q", "kokoro-onnx[gpu]"], check=True)
        subprocess.run(["pip", "install", "-q", "pydub"], check=True)
        open(".deps_installed", "w").close()

    from core.cloner import KokoClone

    cloner = KokoClone()

    out_local = "/tmp/koko_out.wav"

    cloner.generate(
        text=text,
        lang=lang,
        reference_audio=ref_wav,
        output_path=out_local
    )

    os.chdir("..")
    shutil.copy(out_local, output_path)

    return output_path
