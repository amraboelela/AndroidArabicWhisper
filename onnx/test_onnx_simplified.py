#!/usr/bin/env python3
"""
Test the ONNX version of the custom whisper model
"""

import os
import numpy as np
import onnxruntime as ort
import torchaudio
import torch
from transformers import WhisperProcessor
import json

def load_onnx_model(model_dir):
    """Load ONNX encoder and decoder"""
    print(f"Loading ONNX model from {model_dir}...")

    encoder_path = os.path.join(model_dir, "encoder_model.onnx")
    decoder_path = os.path.join(model_dir, "decoder_model.onnx")

    # Create ONNX Runtime sessions
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    print(f"  ✓ Encoder loaded: {encoder_path}")
    print(f"  ✓ Decoder loaded: {decoder_path}")

    return encoder_session, decoder_session

def transcribe_audio_onnx(encoder_session, decoder_session, processor, audio_path):
    """Transcribe audio using ONNX model"""
    print(f"\nTranscribing: {audio_path}")

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Convert to numpy
    audio_array = waveform.squeeze().numpy()

    print(f"  Audio: {len(audio_array) / sample_rate:.2f}s, {sample_rate}Hz")

    # Process audio with WhisperProcessor
    input_features = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features

    # Convert to numpy for ONNX
    input_features_np = input_features.numpy()

    print(f"  Input features shape: {input_features_np.shape}")

    # Run encoder
    print("  Running encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {"input_features": input_features_np}
    )
    encoder_hidden_states = encoder_outputs[0]

    print(f"  Encoder output shape: {encoder_hidden_states.shape}")

    # Initialize decoder with proper Whisper prefix tokens
    decoder_start_token_id = 50258  # <|startoftranscript|>
    lang_token_id = 50272  # <|ar|> (Arabic)
    task_token_id = 50359  # <|transcribe|>
    no_timestamps_token_id = 50363  # <|notimestamps|>
    eos_token_id = 50257  # <|endoftext|>

    print("  Running decoder (greedy decoding)...")

    # Start with proper prefix: <|startoftranscript|><|ar|><|transcribe|><|notimestamps|>
    generated_tokens = [decoder_start_token_id, lang_token_id, task_token_id, no_timestamps_token_id]
    decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    max_length = 200

    for _ in range(max_length):
        # Run decoder
        decoder_outputs = decoder_session.run(
            None,
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states
            }
        )

        logits = decoder_outputs[0]  # Shape: [batch, seq_len, vocab_size]

        # Get next token (greedy)
        next_token = np.argmax(logits[0, -1, :])

        # Stop if EOS token
        if next_token == eos_token_id:
            break

        generated_tokens.append(int(next_token))

        # Update decoder input
        decoder_input_ids = np.array([generated_tokens], dtype=np.int64)

    print(f"  Generated {len(generated_tokens)} tokens")

    # Decode tokens
    transcription = processor.decode(generated_tokens, skip_special_tokens=True)

    return transcription

def main():
    print("="*70)
    print("Testing Simplified ONNX Whisper Model")
    print("="*70)

    model_dir = "models/custom-whisper-ar-quran-onnx-simplified"

    # Load processor (for audio preprocessing and token decoding)
    print("\nLoading processor...")
    processor = WhisperProcessor.from_pretrained(model_dir)
    print("  ✓ Processor loaded")

    # Load ONNX models
    print()
    encoder_session, decoder_session = load_onnx_model(model_dir)

    # Test with audio files
    test_files = [
        "datasets/base/audio/002-04-02.wav",
        "datasets/base/audio/002-04-03.wav",
        "datasets/base/audio/002-04-04.wav",
        "datasets/base/audio/002-04-05.wav",
        "datasets/base/audio/002-04-06.wav",
    ]

    print("\n" + "="*70)
    print("Testing with audio files")
    print("="*70)

    for audio_file in test_files:
        if os.path.exists(audio_file):
            try:
                transcription = transcribe_audio_onnx(
                    encoder_session,
                    decoder_session,
                    processor,
                    audio_file
                )
                print(f"\n  Result: {transcription}")
                print()
            except Exception as e:
                print(f"\n  Error: {e}")
                import traceback
                traceback.print_exc()
                print()
        else:
            print(f"\n  File not found: {audio_file}")

    print("="*70)
    print("✓ ONNX Model testing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
