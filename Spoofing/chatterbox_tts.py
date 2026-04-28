import os
import subprocess
import torch
import torchaudio


def install_dependencies():
    def run(cmd, desc=""):
        print("Installing:", desc)
        subprocess.run(cmd, shell=True, check=False)

    run("pip install --upgrade pip", "pip")
    run("pip uninstall -y chatterbox-tts resemble-perth torchvision protobuf", "remove conflicts")
    run("pip install torch==2.5.0 torchaudio==2.5.0", "torch")
    run("pip install transformers==4.46.3 diffusers==0.29.0", "transformers/diffusers")
    run("pip install huggingface_hub accelerate librosa==0.11.0 safetensors soundfile scipy", "libs")
    run("pip install resemble-perth s3tokenizer conformer", "extra libs")
    run("pip install chatterbox-tts --no-deps", "chatterbox")
    run("pip install protobuf==3.20.3", "protobuf")
    print("✅ Installation complete. Restart runtime now.")


class ChatterboxConfig:
    def __init__(self):
        self.exaggeration = 0.6
        self.cfg_weight = 0.4
        self.max_chunk_words = 50
        self.voice_sample_path = None


def load_model():
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model on:", device)

    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception as e:
        print("GPU failed, loading on CPU:", e)
        model = ChatterboxTTS.from_pretrained(device="cpu")

    print("✅ Model ready")
    return model


def split_into_chunks(text, max_words=50):
    sentences = text.strip().replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sentence in sentences:
        words = sentence.split()

        if len((current + " " + sentence).split()) > max_words and current:
            chunks.append(current.strip() + ".")
            current = sentence
        else:
            current = current + ". " + sentence if current else sentence

    if current:
        chunks.append(current.strip() + ".")

    return chunks


def generate_speech(text, config, model, output_filename="cloned_voice.wav", drive_path=None):
    print("STARTING SPEECH GENERATION")

    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("Voice sample not found")

    chunks = split_into_chunks(text, config.max_chunk_words)
    print("Words:", len(text.split()), "| Chunks:", len(chunks))
    print("Settings:", "exaggeration=", config.exaggeration, "cfg_weight=", config.cfg_weight)
    print("Voice sample:", config.voice_sample_path)

    wav_tensors = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}:", chunk[:80])

        wav = model.generate(
            text=chunk,
            exaggeration=config.exaggeration,
            cfg_weight=config.cfg_weight,
            audio_prompt_path=config.voice_sample_path
        )

        wav_tensors.append(wav)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not wav_tensors:
        print("No audio generated")
        return None, None

    full_audio = torch.cat(wav_tensors, dim=1)

    local_path = "/content/" + output_filename
    torchaudio.save(local_path, full_audio.cpu(), model.sr)

    if drive_path:
        os.makedirs(drive_path, exist_ok=True)
        drive_output = os.path.join(drive_path, output_filename)
        torchaudio.save(drive_output, full_audio.cpu(), model.sr)
        print("✅ Saved to Drive:", drive_output)
        return drive_output, local_path

    print("✅ Saved:", local_path)
    return local_path, local_path


def play_and_analyze_audio(audio_path):
    import IPython.display as ipd
    import matplotlib.pyplot as plt

    if not audio_path or not os.path.exists(audio_path):
        print("Audio file not found:", audio_path)
        return

    ipd.display(ipd.Audio(audio_path))

    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sr

    print("Duration:", round(duration, 2), "seconds")
    print("Sample rate:", sr)

    plt.figure(figsize=(12, 4))
    plt.plot(waveform[0].numpy())
    plt.title("Cloned Voice Waveform")
    plt.grid(True, alpha=0.3)
    plt.show()
