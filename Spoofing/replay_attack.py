# -*- coding: utf-8 -*-
# This module simulates replay attacks on audio files by passing a clean speech signal
# through a chain of speaker, room acoustics, microphone, and noise models.
# Three configurations are provided, ranging from high-quality to low-quality replay conditions.

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from IPython.display import Audio

DEFAULT_SR = 16000

# Loads an audio file and resamples it to the given sample rate.
def load_audio(file_path, sr=DEFAULT_SR):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio, sr

# Applies a bandpass Butterworth filter to simulate the frequency response of a speaker.
def speaker_filter(audio, sr, low_hz, high_hz):
    b, a = signal.butter(4, [low_hz/(sr/2), high_hz/(sr/2)], btype='band')
    return signal.lfilter(b, a, audio)

# Generates a synthetic Room Impulse Response (RIR) with a direct sound and decaying reflections.
def generate_rir(sr, duration, reflection_gain, decay_coef, noise_std):
    t = np.linspace(0, duration, int(sr * duration))
    direct = np.zeros(len(t))
    direct[0] = 1.0
    reflections = reflection_gain * np.exp(-decay_coef * t) * np.random.normal(1, noise_std, len(t))
    rir = direct + reflections
    rir /= np.max(np.abs(rir))
    return rir

# Applies a lowpass Butterworth filter to simulate the frequency response of a microphone.
def mic_filter(audio, sr, cutoff_hz):
    b, a = signal.butter(4, cutoff_hz/(sr/2), btype='low')
    return signal.lfilter(b, a, audio)

# Adds Gaussian white noise to the audio at the specified signal-to-noise ratio (SNR) in dB.
def add_noise(audio, snr_db):
    noise = np.random.normal(0, 1, len(audio))
    audio_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    factor = np.sqrt(audio_power / (10**(snr_db/10) * noise_power))
    return audio + noise * factor

# Simulates a high-quality replay: speaker 300–4000 Hz, short reverb (0.15s),
# weak reflections (0.3), fast decay, mic lowpass 6000 Hz, low noise SNR=35 dB.
def config1(audio, sr=DEFAULT_SR):
    audio_spk = speaker_filter(audio, sr, low_hz=300, high_hz=4000)
    rir = generate_rir(sr, duration=0.15, reflection_gain=0.3, decay_coef=12, noise_std=0.05)
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room, sr, cutoff_hz=6000)
    audio_replay = add_noise(audio_mic, snr_db=35)
    return audio_replay

# Simulates a medium-quality replay: speaker 350–3800 Hz, moderate reverb (0.25s),
# medium reflections (0.5), medium decay, mic lowpass 5000 Hz, moderate noise SNR=25 dB.
def config2(audio, sr=DEFAULT_SR):
    audio_spk = speaker_filter(audio, sr, low_hz=350, high_hz=3800)
    rir = generate_rir(sr, duration=0.25, reflection_gain=0.5, decay_coef=8, noise_std=0.1)
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room, sr, cutoff_hz=5000)
    audio_replay = add_noise(audio_mic, snr_db=25)
    return audio_replay

# Simulates a low-quality replay: speaker 400–3500 Hz, long reverb (0.35s),
# strong reflections (0.7), slow decay, mic lowpass 4000 Hz, heavy noise SNR=15 dB.
def config3(audio, sr=DEFAULT_SR):
    audio_spk = speaker_filter(audio, sr, low_hz=400, high_hz=3500)
    rir = generate_rir(sr, duration=0.35, reflection_gain=0.7, decay_coef=6, noise_std=0.15)
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room, sr, cutoff_hz=4000)
    audio_replay = add_noise(audio_mic, snr_db=15)
    return audio_replay


REPLAY_ENGINES = {
    "replay_c1": config1,
    "replay_c2": config2,
    "replay_c3": config3
}


# Loads a single audio file, applies the selected replay config, and returns the result as float32.
def run_replay_on_file(file_path, config_name, sr=DEFAULT_SR):
    audio, sr = load_audio(file_path, sr)
    engine = REPLAY_ENGINES[config_name]
    output_audio = engine(audio, sr)
    return output_audio.astype(np.float32)

