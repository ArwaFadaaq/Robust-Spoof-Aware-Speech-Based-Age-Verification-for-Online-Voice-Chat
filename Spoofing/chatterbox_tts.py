# -*- coding: utf-8 -*-
import os
import torch
import torchaudio

# =====================================================
# Text Processing
# =====================================================

def split_into_chunks(text, max_words=100):
    sentences = text.strip().replace('\n', ' ').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current_chunk = ""
    current_word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if current_word_count + len(sentence_words) > max_words and current_chunk:
            chunks.append(current_chunk.strip() + ".")
            current_chunk = sentence
            current_word_count = len(sentence_words)
        else:
            current_chunk += (". " if current_chunk else "") + sentence
            current_word_count += len(sentence_words)
    if current_chunk:
        chunks.append(current_chunk.strip() + ".")
    return chunks

def estimate_processing_time(text, words_per_minute=150):
    word_count = len(text.split())
    return word_count, word_count / words_per_minute

# =====================================================
# Config
# =====================================================

class ChatterboxConfig:
    def __init__(self):
        self.exaggeration      = 0.5
        self.cfg_weight        = 0.5
        self.max_chunk_words   = 50
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
        return presets.get(preset_name, presets["neutral"])

# =====================================================
# Model
# =====================================================

def load_model():
    from chatterbox.tts import ChatterboxTTS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on: {device}")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Model loaded")
        return model
    except Exception as e:
        print(f"❌ Failed: {e} — trying CPU...")
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("✅ Model loaded on CPU")
        return model

# =====================================================
# Generation
# =====================================================

def generate_speech(text, config, model, output_path="/content/output.wav"):
    if not config.voice_sample_path or not os.path.exists(config.voice_sample_path):
        raise ValueError("❌ voice_sample_path غير موجود، حدد المسار أولاً")

    chunks = split_into_chunks(text, config.max_chunk_words)
    word_count, time_est = estimate_processing_time(text)

    print(f"📝 {word_count} words | {len(chunks)} chunks | ~{time_est:.1f} min")
    print(f"🎤 Voice: {config.voice_sample_path}")
    print(f"🎛️  exaggeration={config.exaggeration} | cfg_weight={config.cfg_weight}\n")

    wav_tensors = []
    for i, chunk in enumerate(chunks):
        print(f"  chunk {i+1}/{len(chunks)}: {chunk[:60]}...")
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
            print(f"  ❌ Error: {e}")

    if not wav_tensors:
        print("❌ لم يتم توليد أي صوت")
        return None

    full_audio = torch.cat(wav_tensors, dim=1)
    torchaudio.save(output_path, full_audio, model.sr)
    print(f"\n✅ Saved: {output_path}")
    return output_path
