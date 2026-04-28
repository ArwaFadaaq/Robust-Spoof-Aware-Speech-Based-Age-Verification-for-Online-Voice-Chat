import os
import torch
import torchaudio
import numpy as np
import soundfile as sf

DEFAULT_SR = 16000
NOISE_TYPES = ["gaussian", "cafe", "environmental"]
NOISE_LEVELS = {"clean": None, "low": 20, "med": 10, "hard": 5}


def load_audio(path, target_sr=DEFAULT_SR):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav / wav.abs().max().clamp_min(1e-9)
    return wav.squeeze(0), target_sr


def load_noise_files(cafe_path, environmental_path, sr=DEFAULT_SR):
    cafe_wav, _ = load_audio(cafe_path, target_sr=sr)
    environmental_wav, _ = load_audio(environmental_path, target_sr=sr)
    print(" cafe loaded: " + str(round(cafe_wav.shape[0] / sr, 2)) + "s")
    print(" environmental loaded: " + str(round(environmental_wav.shape[0] / sr, 2)) + "s")
    return {"cafe": cafe_wav, "environmental": environmental_wav}


def apply_noise(waveform, seed, config):
    snr_db_min = float(config.get("snr_db_min", 5))
    snr_db_max = float(config.get("snr_db_max", 20))
    g = torch.Generator(device=waveform.device).manual_seed(int(seed))
    snr_db = torch.empty((), device=waveform.device).uniform_(snr_db_min, snr_db_max, generator=g).item()
    sig_power = waveform.pow(2).mean()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn(waveform.numel(), device=waveform.device, generator=g).view_as(waveform) * noise_power.sqrt()
    return (waveform + noise).clamp_(-1, 1)


def apply_background_mix(waveform, background, seed, config):
    snr_db_min = float(config.get("snr_db_min", 0))
    snr_db_max = float(config.get("snr_db_max", 15))
    background = background.to(device=waveform.device, dtype=waveform.dtype)
    g = torch.Generator(device=waveform.device).manual_seed(int(seed))
    snr_db = torch.empty((), device=waveform.device).uniform_(snr_db_min, snr_db_max, generator=g).item()

    def to_ct(x):
        return x.unsqueeze(0) if x.ndim == 1 else x

    x = to_ct(waveform)
    n = to_ct(background)

    C, T = x.shape
    Cn, Tn = n.shape

    if Cn == 1 and C > 1:
        n = n.expand(C, -1)
    elif Cn != C:
        raise ValueError("Channel mismatch")

    if Tn < T:
        n = n.repeat(1, (T + Tn - 1) // Tn)[:, :T]
    elif Tn > T:
        start = int(torch.randint(0, Tn - T + 1, (1,), generator=g, device=waveform.device).item())
        n = n[:, start:start + T]

    scale = (x.pow(2).mean() / (10 ** (snr_db / 10)) / n.pow(2).mean().clamp_min(1e-12)).sqrt()
    y = x + n * scale

    if waveform.ndim == 1:
        y = y.squeeze(0)

    return y.clamp_(float(config.get("clamp_min", -1)), float(config.get("clamp_max", 1)))


def pick_noise_level(noise_probs):
    return np.random.choice(list(noise_probs.keys()), p=list(noise_probs.values()))


def pick_noise_type():
    return NOISE_TYPES[np.random.randint(0, len(NOISE_TYPES))]


def apply_noise_to_segment(segment, noise_buffers, noise_probs, seed=None):
    if seed is None:
        seed = int(np.random.randint(0, 99999))

    level = pick_noise_level(noise_probs)

    if level == "clean":
        return segment, {"level": "clean", "noise_type": None, "snr_db": None, "seed": seed}

    noise_type = pick_noise_type()
    snr_db = NOISE_LEVELS[level]
    config = {"snr_db_min": snr_db, "snr_db_max": snr_db}

    if noise_type == "gaussian":
        noisy = apply_noise(segment, seed, config)
    else:
        noisy = apply_background_mix(segment, noise_buffers[noise_type], seed, config)

    return noisy, {"level": level, "noise_type": noise_type, "snr_db": snr_db, "seed": seed}


def run_noise_on_file(file_path, noise_buffers, noise_probs, sr=DEFAULT_SR):
    audio, sr = load_audio(file_path, target_sr=sr)
    print(" " + file_path)
    print("   Duration: " + str(round(len(audio) / sr, 2)) + "s")

    seed = int(np.random.randint(0, 99999))

    noisy_audio, info = apply_noise_to_segment(
        audio,
        noise_buffers,
        noise_probs,
        seed=seed
    )

    print("level: " + info["level"] + " | type: " + str(info["noise_type"]) + " | SNR: " + str(info["snr_db"]))

    return noisy_audio, info


def run_noise_on_folder(input_folder, output_folder, noise_buffers, noise_probs, sr=DEFAULT_SR):
    os.makedirs(output_folder, exist_ok=True)
    all_results = {}

    for file in os.listdir(input_folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            input_path = os.path.join(input_folder, file)

            noisy_audio, info = run_noise_on_file(
                input_path,
                noise_buffers,
                noise_probs,
                sr
            )

            out_path = os.path.join(output_folder, os.path.splitext(file)[0] + "_noisy.wav")
            sf.write(out_path, noisy_audio.numpy(), sr)
            all_results[file] = info

    return all_results
