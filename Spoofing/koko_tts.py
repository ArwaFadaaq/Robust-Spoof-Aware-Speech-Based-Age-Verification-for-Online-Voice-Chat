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
    cloner = KokoClone()
    out_local = "/tmp/koko_out.wav"
    cloner.generate(
        text=text,
        lang=lang,
        reference_audio=reference_audio_path,
        output_path=out_local
    )
    os.chdir("..")
    shutil.copy(out_local, output_path)
    return output_path
