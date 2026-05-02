import os
import torch
import torchaudio
import numpy as np
import soundfile as sf

DEFAULT_SR = 16000
NOISE_LEVELS = {"clean": None, "low": 20, "med": 10, "hard": 5}
NOISE_PROBS = {"clean": 0.50, "low": 0.25, "med": 0.20, "hard": 0.05}


def load_audio(path, target_sr=DEFAULT_SR):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav / wav.abs().max().clamp_min(1e-9)
    return wav.squeeze(0), target_sr


def apply_noise(waveform, seed, config):
    snr_db_min = float(config.get("snr_db_min", 5))
    snr_db_max = float(config.get("snr_db_max", 20))
    g = torch.Generator(device=waveform.device).manual_seed(int(seed))
    snr_db = torch.empty((), device=waveform.device).uniform_(snr_db_min, snr_db_max, generator=g).item()
    sig_power = waveform.pow(2).mean()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn(waveform.numel(), device=waveform.device, generator=g).view_as(waveform) * noise_power.sqrt()
    return (waveform + noise).clamp_(-1, 1)


def pick_noise_level():
    return np.random.choice(list(NOISE_PROBS.keys()), p=list(NOISE_PROBS.values()))


def apply_noise_to_segment(segment, seed=None):
    if seed is None:
        seed = int(np.random.randint(0, 99999))

    level = pick_noise_level()

    if level == "clean":
        return segment, {"level": "clean", "noise_type": None, "snr_db": None, "seed": seed}

    snr_db = NOISE_LEVELS[level]
    config = {"snr_db_min": snr_db, "snr_db_max": snr_db}
    noisy = apply_noise(segment, seed, config)

    return noisy, {"level": level, "noise_type": "gaussian", "snr_db": snr_db, "seed": seed}


def run_noise_on_file(file_path, sr=DEFAULT_SR):
    audio, sr = load_audio(file_path, target_sr=sr)
    print(" " + file_path)
    print("   Duration: " + str(round(len(audio) / sr, 2)) + "s")

    seed = int(np.random.randint(0, 99999))
    noisy_audio, info = apply_noise_to_segment(audio, seed=seed)

    print("level: " + info["level"] + " | type: " + str(info["noise_type"]) + " | SNR: " + str(info["snr_db"]))

    return noisy_audio, info


def run_noise_on_folder(input_folder, output_folder, sr=DEFAULT_SR):
    os.makedirs(output_folder, exist_ok=True)
    all_results = {}

    for file in os.listdir(input_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            input_path = os.path.join(input_folder, file)
            noisy_audio, info = run_noise_on_file(input_path, sr)

            out_path = os.path.join(output_folder, os.path.splitext(file)[0] + "_noisy.wav")
            sf.write(out_path, noisy_audio.numpy(), sr)

            all_results[file] = info

    return all_results
