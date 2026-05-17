# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from IPython.display import Audio

DEFAULT_SR = 16000

def load_audio(file_path, sr=DEFAULT_SR):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio, sr

def config1(audio, sr=DEFAULT_SR):

    def speaker_filter(audio):
        b, a = signal.butter(4, [300/(sr/2), 4000/(sr/2)], btype='band')
        return signal.lfilter(b, a, audio)

    def generate_rir(duration=0.15):
        t = np.linspace(0, duration, int(sr * duration))
        direct = np.zeros(len(t))
        direct[0] = 1.0
        reflections = 0.3 * np.exp(-12 * t) * np.random.normal(1, 0.05, len(t))
        rir = direct + reflections
        rir /= np.max(np.abs(rir))
        return rir

    def mic_filter(audio):
        b, a = signal.butter(4, 6000/(sr/2), btype='low')
        return signal.lfilter(b, a, audio)

    def add_noise(audio, snr_db=35):
        noise = np.random.normal(0, 1, len(audio))
        audio_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)
        factor = np.sqrt(audio_power / (10**(snr_db/10) * noise_power))
        return audio + noise * factor

    audio_spk = speaker_filter(audio)
    rir = generate_rir()
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room)
    audio_replay = add_noise(audio_mic, 35)

    return audio_replay


def config2(audio, sr=DEFAULT_SR):

    def speaker_filter(audio):
        b, a = signal.butter(4, [150/(sr/2), 5000/(sr/2)], btype='band')
        return signal.lfilter(b, a, audio)

    def generate_rir(duration=0.25):
        t = np.linspace(0, duration, int(sr * duration))
        direct = np.zeros(len(t))
        direct[0] = 1.0
        reflections = 0.5 * np.exp(-8 * t) * np.random.normal(1, 0.1, len(t))
        rir = direct + reflections
        rir /= np.max(np.abs(rir))
        return rir

    def mic_filter(audio):
        b, a = signal.butter(4, 5000/(sr/2), btype='low')
        return signal.lfilter(b, a, audio)

    def add_noise(audio, snr_db=25):
        noise = np.random.normal(0, 1, len(audio))
        audio_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)
        factor = np.sqrt(audio_power / (10**(snr_db/10) * noise_power))
        return audio + noise * factor

    audio_spk = speaker_filter(audio)
    rir = generate_rir()
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room)
    audio_replay = add_noise(audio_mic, 25)

    return audio_replay


def config3(audio, sr=DEFAULT_SR):

    def speaker_filter(audio):
        b, a = signal.butter(4, [400/(sr/2), 3500/(sr/2)], btype='band')
        return signal.lfilter(b, a, audio)

    def generate_rir(duration=0.35):
        t = np.linspace(0, duration, int(sr * duration))
        direct = np.zeros(len(t))
        direct[0] = 1.0
        reflections = 0.7 * np.exp(-6 * t) * np.random.normal(1, 0.15, len(t))
        rir = direct + reflections
        rir /= np.max(np.abs(rir))
        return rir

    def mic_filter(audio):
        b, a = signal.butter(4, 4000/(sr/2), btype='low')
        return signal.lfilter(b, a, audio)

    def add_noise(audio, snr_db=15):
        noise = np.random.normal(0, 1, len(audio))
        audio_power = np.mean(audio**2)
        noise_power = np.mean(noise**2)
        factor = np.sqrt(audio_power / (10**(snr_db/10) * noise_power))
        return audio + noise * factor

    audio_spk = speaker_filter(audio)
    rir = generate_rir()
    audio_room = signal.convolve(audio_spk, rir, mode='same')
    audio_mic = mic_filter(audio_room)
    audio_replay = add_noise(audio_mic, 15)

    return audio_replay


REPLAY_ENGINES = {
    "replay_c1": config1,
    "replay_c2": config2,
    "replay_c3": config3
}


def run_replay_on_file(file_path, config_name, sr=DEFAULT_SR):
    audio, sr = load_audio(file_path, sr)
    engine = REPLAY_ENGINES[config_name]
    output_audio = engine(audio, sr)
    return output_audio.astype(np.float32)


def run_replay_on_folder(input_folder, output_folder, config_name, sr=DEFAULT_SR):
    os.makedirs(output_folder, exist_ok=True)
    output_files = []
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            audio, sr = load_audio(input_path, sr)
            engine = REPLAY_ENGINES[config_name]
            output_audio = engine(audio, sr)
            sf.write(output_path, output_audio, sr)
            output_files.append(output_path)
    return output_files
