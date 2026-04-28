# chatterbox_tts.py
# Chatterbox TTS + Voice Clone — reusable module
# Place this file in: Spoofing/chatterbox_tts.py

import os
import sys
import subprocess
import torch
import torchaudio


# ─── Installation ─────────────────────────────────────────────────────────────

def install_dependencies():
    def run(cmd, desc=""):
        print("Installing: " + desc)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Warning: " + desc + " failed")
            print(result.stderr)
            return False
        print("Done: " + desc)
        return True

    run("pip install --upgrade pip", "pip upgrade")
    run("pip uninstall -y chatterbox-tts resemble-perth", "removing conflicts")
    run("pip install torch==2.5.0 torchaudio==2.5.0", "PyTorch 2.5.0")
    run("pip install transformers==4.46.3", "transformers")
    run("pip install diffusers==0.29.0", "diffusers")
    run("pip install huggingface_hub>=0.23.0", "huggingface_hub")
    run("pip install accelerate>=0.25.0", "accelerate")
    run("apt update && apt install -y git-lfs", "git-lfs")
    run("pip install 'numpy>=1.24.0,<1.26.0' librosa==0.11.0 safetensors soundfile scipy", "audio libs")
    run("pip install resemble-perth", "resemble-perth")
    run("pip install s3tokenizer conformer", "s3tokenizer + conformer")
    run("pip install chatterbox-tts --no-deps", "chatterbox-tts")
    run("pip uninstall -y protobuf", "remove protobuf")
    run("pip install protobuf==3.20.3", "protobuf 3.20.3")
    print("Installation complete. Restart runtime if needed.")


# ─── Config ───────────────────────────────────────────────────────────────────

class ChatterboxConfig:
    def __init__(self):
        self.exaggeration = 0.5
        self.cfg_weight = 0.5
        self.max_chunk_words = 50
        self.voice_sample_path = None

    def get_preset(self, preset_name):
        presets = {
            "neutral":     {"exaggeration": 0.5, "cfg_weight": 0.5},
            "calm":        {"exaggeration": 0.3, "cfg_weight": 0.6},
            "expressive":  {"exaggeration": 0.7, "cfg_weight": 0.4},
            "dramatic":    {"exaggeration": 1.0, "cfg_weight": 0.3},
            "storytelling":{"exaggeration": 0.8, "cfg_weight": 0.4},
            "audiobook":   {"exaggeration": 0.4, "cfg_weight": 0.6},
            "fast_speaker":{"exaggeration": 0.6, "cfg_weight": 0.3},
        }
        p = presets.get(preset_name, presets["neutral"])
        self.exaggeration = p["exaggeration"]
        self.cfg_weight = p["cfg_weight"]
        return p


# ─── Model ────────────────────────────────────────────────────────────────────

def load_model():
    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model on: " + device)
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("GPU failed, trying CPU... " + str(e))
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("Model loaded on CPU")
        return model


# ─── Text Processing ──────────────────────────────────────────────────────────

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


def estimate_processing_time(text, words_per_minute=150):
    word_count = len(text.split())
    return word_count, word_count / words_per_minute


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_speech(text, config, model, output_filename="generated_speech.wav", drive_path=None):
    print("STARTING SPEECH GENERATION")

    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("Voice sample not found. Set config.voice_sample_path first.")

    chunks = split_into_chunks(text, config.max_chunk_words)
    word_count, time_est = estimate_processing_time(text)

    print("Words: " + str(word_count) + " | Chunks: " + str(len(chunks)) + " | Est. time: " + str(round(time_est, 1)) + " min")
    print("Settings: exaggeration=" + str(config.exaggeration) + ", cfg_weight=" + str(config.cfg_weight))
    print("Voice sample: " + config.voice_sample_path)

    wav_tensors = []
    for i, chunk in enumerate(chunks):
        print("Chunk " + str(i + 1) + "/" + str(len(chunks)) + ": " + chunk[:50])
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
            print("Error in chunk " + str(i + 1) + ": " + str(e))

    if not wav_tensors:
        print("No audio generated")
        return None, None

    full_audio = torch.cat(wav_tensors, dim=1)
    local_path = "/content/" + output_filename
    torchaudio.save(local_path, full_audio, model.sr)
    print("Saved locally: " + local_path)

    if drive_path:
        drive_out = drive_path + "/" + output_filename
        torchaudio.save(drive_out, full_audio, model.sr)
        print("Saved to Drive: " + drive_out)
        return drive_out, local_path

    return local_path, local_path


# ─── Playback & Analysis ──────────────────────────────────────────────────────

def play_and_analyze_audio(audio_path):
    import IPython.display as ipd
    import matplotlib.pyplot as plt

    if not audio_path or not os.path.exists(audio_path):
        print("Audio file not found: " + str(audio_path))
        return

    print("Playing: " + audio_path)
    ipd.display(ipd.Audio(audio_path))

    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
        print("Duration: " + str(round(duration, 2)) + "s | SR: " + str(sr) + "Hz | Channels: " + str(waveform.shape[0]))
        plt.figure(figsize=(12, 4))
        plt.plot(waveform[0].numpy())
        plt.title("Cloned Voice Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.show()
    except Exception as e:
        print("Analysis error: " + str(e))


# ─── Experiments ──────────────────────────────────────────────────────────────

def experiment_with_parameters(text, config, model):
    import IPython.display as ipd

    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("Run voice sample setup first.")

    experiments = [
        {"name": "Neutral",    "exaggeration": 0.5, "cfg_weight": 0.5},
        {"name": "Expressive", "exaggeration": 0.8, "cfg_weight": 0.4},
        {"name": "Calm",       "exaggeration": 0.3, "cfg_weight": 0.6},
        {"name": "Dramatic",   "exaggeration": 1.0, "cfg_weight": 0.3},
    ]

    print("PARAMETER EXPERIMENTS")
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
