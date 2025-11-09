#!/usr/bin/env python3
"""
Test ONNX models with sample audio
"""
import sys
import onnxruntime as ort
import numpy as np
import json
import torchaudio
import torch
from pathlib import Path

def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract mel features (same as training)"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Pad or trim to EXACTLY 30 seconds (480000 samples)
    # This ensures exactly 3000 mel frames after STFT
    target_length = 480000
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        padded = torch.zeros(1, target_length)
        padded[:, :waveform.shape[1]] = waveform
        waveform = padded

    # Extract mel spectrogram with exact Whisper parameters
    # n_fft=400, hop_length=160 → 480000/160 = 3000 frames exactly
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        f_min=0,
        f_max=8000
    )
    mel_spec = mel_transform(waveform)

    # Ensure exactly 3000 frames (trim if needed due to padding effects)
    if mel_spec.shape[2] > 3000:
        mel_spec = mel_spec[:, :, :3000]

    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    # Global Whisper normalization
    mel_mean = -4.2677
    mel_std = 4.5689
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    # Transpose to (batch=1, n_mels=80, time)
    mel_features = mel_features.transpose(0, 1).unsqueeze(0).numpy()

    return mel_features

def main():
    print("Testing ONNX models...")

    # Paths
    encoder_path = "models/onnx/encoder_model.onnx"
    decoder_path = "models/onnx/decoder_model.onnx"
    vocab_path = "models/onnx/vocabulary.json"
    audio_path = "datasets/Quran-A/audio/001/001-001.wav"

    # Check files exist
    if not Path(encoder_path).exists():
        print(f"❌ Encoder not found: {encoder_path}")
        return
    if not Path(decoder_path).exists():
        print(f"❌ Decoder not found: {decoder_path}")
        return

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"✓ Vocabulary loaded: {len(vocab)} words")

    # Load ONNX sessions
    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(decoder_path, providers=['CPUExecutionProvider'])
    print("✓ ONNX models loaded")

    # Extract audio features
    print(f"Loading audio: {audio_path}")
    mel_features = extract_mel_features(audio_path)
    print(f"✓ Mel features shape: {mel_features.shape}")

    # Run encoder
    print("Running encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {"input_features": mel_features.astype(np.float32)}
    )
    encoder_hidden_states = encoder_outputs[0]
    print(f"✓ Encoder output shape: {encoder_hidden_states.shape}")

    # Run decoder with greedy decoding
    print("Running decoder...")
    generated_tokens = []
    max_tokens = 50
    SOS_TOKEN = 1
    EOS_TOKEN = 2

    for step in range(max_tokens):
        # Build input_ids: [SOS] + generated tokens so far
        input_ids = np.array([[SOS_TOKEN] + generated_tokens], dtype=np.int64)

        # Run decoder
        decoder_outputs = decoder_session.run(
            None,
            {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states.astype(np.float32)
            }
        )
        logits = decoder_outputs[0]

        # Get next token (greedy)
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == EOS_TOKEN:
            break

        generated_tokens.append(next_token)

    print(f"✓ Generated {len(generated_tokens)} tokens")

    # Decode tokens to text
    generated_words = [vocab[idx] if idx < len(vocab) else "<unk>" for idx in generated_tokens]
    result = " ".join(generated_words)

    print("\n" + "="*60)
    print("Transcription Result:")
    print("="*60)
    print(result)
    print("="*60)

    # Expected text for 001-001
    expected = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    print(f"\nExpected: {expected}")
    print(f"Generated: {result}")

    if result.strip() == expected.strip():
        print("\n✓ Perfect match!")
    else:
        print("\n⚠️  Output doesn't match expected")

if __name__ == "__main__":
    main()
