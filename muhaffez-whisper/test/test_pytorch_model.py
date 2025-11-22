#!/usr/bin/env python3
"""Test PyTorch model directly"""
import sys
sys.path.append('../tools')

import torch
import torchaudio
import json
from encoder_decoder_transformer import EncoderDecoderTransformer

# Load vocabulary
with open('../models/vocabulary.json', 'r') as f:
    vocab = json.load(f)
vocab_size = len(vocab)
print(f'✅ Vocabulary loaded: {vocab_size} words')

# Load PyTorch model
model = EncoderDecoderTransformer(
    vocab_size=vocab_size,
    d_model=128,
    n_encoder_layers=4,
    n_decoder_layers=4,
    n_heads=4,
    d_ff=512,
    dropout=0.0,
    max_seq_len=1500,
    n_mels=40
)
checkpoint = torch.load('../models/muhaffez_whisper.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
print('✅ PyTorch model loaded')

# Load audio
audio_path = '../datasets/Quran-A/audio/001/001-003.wav'
waveform, sample_rate = torchaudio.load(audio_path)
print(f'✅ Audio loaded: {waveform.shape[1]} samples at {sample_rate}Hz')

# Extract mel features (same as training)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    hop_length=160,
    n_mels=40,
    f_min=0,
    f_max=8000
)
mel_spec = mel_transform(waveform)
mel_spec = torch.log(mel_spec + 1e-9)
mel_features = mel_spec.squeeze(0).transpose(0, 1)

# Global normalization
mel_mean = -4.2677
mel_std = 4.5689
mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

print(f'✅ Mel features: {mel_features.shape}')

# Run PyTorch model
with torch.no_grad():
    # Encode (expects batch, n_mels, time)
    mel_input = mel_features.transpose(0, 1).unsqueeze(0)
    encoder_output = model.encode(mel_input)
    print(f'✅ Encoder output: {encoder_output.shape}')

    # Decode
    generated = []
    input_ids = torch.tensor([[1]], dtype=torch.long)  # SOS

    for step in range(10):
        decoder_output, _ = model.decode(input_ids, encoder_output)
        logits = decoder_output[0, -1, :]
        next_token = logits.argmax().item()

        if next_token == 2:  # EOS
            break

        generated.append(next_token)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    print(f'\n🎯 Generated {len(generated)} tokens: {generated}')
    text = ' '.join([vocab[idx] for idx in generated])
    print(f'📝 Text: {text}')
    print(f'\n✨ Expected: بسم الله الرحمن الرحيم (4 words)')
