import os
import subprocess
import torch
import torchaudio


def install_dependencies():
    def run(cmd, desc=""):
        print("Installing: " + desc)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("Done: " + desc if result.returncode == 0 else "Failed: " + desc)
        return result.returncode == 0
    run("pip install --upgrade pip", "pip upgrade")
    run("pip uninstall -y chatterbox-tts resemble-perth torchvision", "removing conflicts")
    run("pip install torch==2.5.0 torchaudio==2.5.0", "PyTorch 2.5.0")
    run("pip install transformers==4.46.3", "transformers")
    run("pip install diffusers==0.29.0", "diffusers")
    run("pip install huggingface_hub>=0.23.0", "huggingface_hub")
    run("pip install accelerate>=0.25.0", "accelerate")
    run("apt update && apt install -y git-lfs", "git-lfs")
    run("pip install librosa==0.11.0 safetensors soundfile scipy", "audio libs")
    run("pip install resemble-perth", "resemble-perth")
    run("pip install s3tokenizer conformer", "s3tokenizer + conformer")
    run("pip install chatterbox-tts --no-deps", "chatterbox-tts")
    run("pip uninstall -y protobuf", "remove protobuf")
    run("pip install protobuf==3.20.3", "protobuf 3.20.3")
    print("Installation complete. Restart runtime.")


class ChatterboxConfig:
    def __init__(self):
        self.exaggeration = 0.5
        self.cfg_weight = 0.5
        self.max_chunk_words = 50
        self.voice_sample_path = None


def load_model():
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model on: " + device)
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("GPU failed, trying CPU: " + str(e))
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("Model loaded on CPU")
        return model


def split_into_chunks(text, max_words=100):
    sentences = text.strip().replace("\n", " ").split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current_chunk = ""
    current_word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > max_words and current_chunk:
            chunks.append(current_chunk.strip() + ".")
            current_chunk = sentence
            current_word_count = len(words)
        else:
            current_chunk = (current_chunk + ". " + sentence) if current_chunk else sentence
            current_word_count += len(words)
    if current_chunk:
        chunks.append(current_chunk.strip() + ".")
    return chunks


def generate_speech(text, config, model, output_filename="generated_speech.wav", drive_path=None):
    print("Starting speech generation")
    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("Voice sample not found. Set config.voice_sample_path first.")
    chunks = split_into_chunks(text, config.max_chunk_words)
    word_count = len(text.split())
    print("Words: " + str(word_count) + " | Chunks: " + str(len(chunks)))
    wav_tensors = []
    for i, chunk in enumerate(chunks):
        print("Chunk " + str(i+1) + "/" + str(len(chunks)) + ": " + chunk[:50])
        try:
            wav = model.generate(
                text=chunk,
                exaggeration=config.exaggeration,
                cfg_weight=config.cfg_weight,
                audio_prompt_path=config.voice_sample_path
            )
            wav_tensors.append(wav)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print("Error in chunk " + str(i+1) + ": " + str(e))
    if not wav_tensors:
        print("No audio generated")
        return None, None
    full_audio = torch.cat(wav_tensors, dim=1)
    local_path = "/content/" + output_filename
    torchaudio.save(local_path, full_audio, model.sr)
    print("Saved: " + local_path)
    if drive_path:
        drive_out = drive_path + "/" + output_filename
        torchaudio.save(drive_out, full_audio, model.sr)
        print("Drive: " + drive_out)
        return drive_out, local_path
    return local_path, local_path


def play_and_analyze_audio(audio_path):
    import IPython.display as ipd
    import matplotlib.pyplot as plt
    if not audio_path or not os.path.exists(audio_path):
        print("File not found: " + str(audio_path))
        return
    ipd.display(ipd.Audio(audio_path))
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sr
    print("Duration: " + str(round(duration, 2)) + "s | SR: " + str(sr) + "Hz")
    plt.figure(figsize=(12, 4))
    plt.plot(waveform[0].numpy())
    plt.title("Cloned Voice Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.show()


def experiment_with_parameters(text, config, model):
    import IPython.display as ipd
    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("Set voice_sample_path first.")
    experiments = [
        {"name": "Neutral",    "exaggeration": 0.5, "cfg_weight": 0.5},
        {"name": "Expressive", "exaggeration": 0.8, "cfg_weight": 0.4},
        {"name": "Calm",       "exaggeration": 0.3, "cfg_weight": 0.6},
        {"name": "Dramatic",   "exaggeration": 1.0, "cfg_weight": 0.3},
    ]
    for exp in experiments:
        print("Testing: " + exp["name"])
        try:
            wav = model.generate(
                text=text,
                exaggeration=exp["exaggeration"],
                cfg_weight=exp["cfg_weight"],
                audio_prompt_path=config.voice_sample_path
            )
            filepath = "/content/experiment_" + exp["name"].lower() + ".wav"
            torchaudio.save(filepath, wav, model.sr)
            print("Done: " + filepath)
            ipd.display(ipd.Audio(filepath))
        except Exception as e:
            print("Error: " + str(e))
