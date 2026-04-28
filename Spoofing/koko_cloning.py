import os
import shutil
import subprocess


def run_koko(
    text: str,
    reference_audio_path: str,
    lang: str = "en",
    output_path: str = "koko_output.wav"
) -> str:

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
    from pydub import AudioSegment

    cloner = KokoClone()

    # convert to WAV if not already
    ref_local = "reference.wav"
    AudioSegment.from_file(f"../{reference_audio_path}")\
        .set_frame_rate(22050)\
        .set_channels(1)\
        .export(ref_local, format="wav")

    out_local = "koko_out.wav"

    cloner.generate(
        text=text,
        lang=lang,
        reference_audio=ref_local,
        output_path=out_local
    )

    os.chdir("..")
    shutil.copy(f"kokoclone/{out_local}", output_path)

    return output_path
