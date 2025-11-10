#!/usr/bin/env python3
"""
Test script for Muhaffez Whisper ONNX model with audio file (offline mode)
Usage: python3 test_muhaffez_model.py [audio_file_path]
       python3 test_muhaffez_model.py /path/to/segment.wav
"""

import os
import sys
import numpy as np
import torch
import torchaudio
import json
import onnxruntime as ort

# Paths
MODEL_DIR = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/muhaffez_whisper"
DEFAULT_AUDIO_PATH = "/Users/amraboelela/develop/android/AndroidArabicWhisper/app/src/main/assets/001.wav"
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model.onnx")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocabulary.json")

# Whisper constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples

def load_vocabulary(vocab_path):
    """Load vocabulary from JSON file"""
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Create reverse mapping (id -> token)
    if isinstance(vocab, dict):
        id_to_token = {v: k for k, v in vocab.items()}
    else:
        id_to_token = {i: t for i, t in enumerate(vocab)}

    return vocab, id_to_token

def load_audio(audio_path, target_sr=16000):
    """Load and preprocess audio file"""
    print(f"Loading audio from: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)

    print(f"Original sample rate: {sample_rate} Hz")
    print(f"Original shape: {waveform.shape}")

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print("Converted to mono")

    # Resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        print(f"Resampled to {target_sr} Hz")

    # Convert to numpy (don't pad - use actual audio length)
    audio_array = waveform.squeeze().numpy()

    print(f"Final audio shape: {audio_array.shape}")
    print(f"Audio duration: {len(audio_array) / target_sr:.2f} seconds")

    return audio_array

def log_mel_spectrogram(audio, n_mels=N_MELS):
    """
    Compute log-Mel spectrogram as expected by Whisper model (matching test_full.py)
    """
    print("\nExtracting mel spectrogram features...")

    # Use torch for mel spectrogram
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Add channel dimension

    # Whisper parameters (100 fps: 16000 / 160 = 100)
    n_fft = 400
    hop_length = 160

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=SAMPLE_RATE // 2,
        # Try without explicit norm/mel_scale to see if model was trained differently
        # window_fn=torch.hann_window,
        # center=True,
        # pad_mode='reflect',
        # norm='slaney',
        # mel_scale='htk'
    )

    mel_spec = mel_transform(audio_tensor)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    # Global Whisper normalization
    WHISPER_MEL_MEAN = -4.2677393
    WHISPER_MEL_STD = 4.5689974
    mel_features = (mel_features - WHISPER_MEL_MEAN) / WHISPER_MEL_STD

    # Transpose back and add batch dimension for ONNX model (batch, n_mels, time)
    input_features = mel_features.transpose(0, 1).unsqueeze(0).numpy()

    print(f"Input features shape: {input_features.shape}")
    return input_features

def decode_tokens(tokens, id_to_token):
    """Decode token IDs to text"""
    # Skip the <s> at the beginning if present
    if tokens and tokens[0] == 1:
        tokens = tokens[1:]

    # Remove </s> at the end if present
    if 2 in tokens:
        tokens = tokens[:tokens.index(2)]

    # Convert token IDs to words
    words = [id_to_token[idx] for idx in tokens if idx in id_to_token and idx not in [0, 1, 2]]

    return ' '.join(words)

def test_with_onnx(audio_path):
    """Test using ONNX models directly"""
    print("="*60)
    print("Testing Muhaffez Whisper ONNX Model (Offline)")
    print("="*60)
    print(f"Audio file: {audio_path}")
    print()

    # Load vocabulary
    vocab, id_to_token = load_vocabulary(VOCAB_PATH)

    # Load audio
    audio_array = load_audio(audio_path)

    # Extract features
    input_features = log_mel_spectrogram(audio_array)

    # Debug: Print some sample values from mel features for comparison
    print(f"\n📊 Mel feature sample values (first 5x5):")
    for i in range(min(5, input_features.shape[1])):
        row_str = " ".join([f"{input_features[0, i, j]:7.3f}" for j in range(min(5, input_features.shape[2]))])
        print(f"  Mel[{i}]: {row_str}")

    print(f"\n📊 Mel feature stats:")
    print(f"  Min: {input_features.min():.4f}")
    print(f"  Max: {input_features.max():.4f}")
    print(f"  Mean: {input_features.mean():.4f}")
    print(f"  Std: {input_features.std():.4f}")

    # Load ONNX models
    print("\nLoading ONNX models...")
    print(f"Encoder: {ENCODER_PATH}")
    print(f"Decoder: {DECODER_PATH}")

    encoder_session = ort.InferenceSession(ENCODER_PATH)
    decoder_session = ort.InferenceSession(DECODER_PATH)

    print("ONNX models loaded successfully!")

    # Run encoder
    print("\nRunning encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {"input_features": input_features}
    )
    encoder_hidden_states = encoder_outputs[0]
    print(f"Encoder output shape: {encoder_hidden_states.shape}")

    # Prepare decoder inputs
    print("\nPreparing decoder inputs...")
    # Muhaffez Whisper special tokens: 0 = <unk>, 1 = <s>, 2 = </s>
    # Start with: <s>
    initial_tokens = [1]  # Start of sequence
    decoder_input_ids = np.array([initial_tokens], dtype=np.int64)

    # Generate tokens
    print("\nGenerating tokens...")
    max_length = 200
    generated_tokens = list(initial_tokens)

    for i in range(max_length):
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_hidden_states
        }

        decoder_outputs = decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]

        # Get next token
        next_token = np.argmax(logits[0, -1, :])
        generated_tokens.append(int(next_token))

        # Check for end token (2 is </s>)
        if next_token == 2:
            print(f"End token found at position {i}")
            break

        # Update decoder input with all tokens so far
        decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    print(f"\nGenerated {len(generated_tokens)} tokens")
    print(f"Token IDs: {generated_tokens}")

    # Decode tokens
    print("\nDecoding tokens...")
    transcription = decode_tokens(generated_tokens, id_to_token)

    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT:")
    print("="*60)
    print(transcription)
    print("="*60)

    return transcription

def main():
    # Get audio path from command line argument or use default
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            return 1
    else:
        audio_path = DEFAULT_AUDIO_PATH
        print(f"No audio file specified, using default: {audio_path}")

    try:
        transcription = test_with_onnx(audio_path)
        print("\nTest completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
