#!/usr/bin/env python3
"""
Test tarteel-ai/whisper-base-ar-quran model with preloaded weights
"""

import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio

def load_model_and_processor():
    """Load the custom tarteel model and processor"""
    model_path = "models/custom-whisper-ar-quran"

    print(f"Loading CUSTOM model from {model_path}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path)

    print(f"✓ Custom model loaded successfully")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"  Config: {model.config}")

    return model, processor

def transcribe_audio(model, processor, audio_path):
    """Transcribe audio file"""
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

    # Convert to numpy and squeeze
    audio_array = waveform.squeeze().numpy()

    print(f"  Audio: {len(audio_array) / sample_rate:.2f}s, {sample_rate}Hz")

    # Process audio
    input_features = processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features

    # Generate transcription
    print("  Generating transcription...")

    model.eval()
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            max_new_tokens=400,  # Maximum new tokens (less than max_target_positions 448)
            num_beams=5          # Beam search for better quality
        )

    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription

def main():
    print("="*60)
    print("Testing Custom Whisper Model (models/custom-whisper-ar-quran)")
    print("="*60)

    # Load model
    model, processor = load_model_and_processor()

    # Test with an audio file
    audio_file = "datasets/base/audio/002-04-01.wav"

    if not os.path.exists(audio_file):
        print(f"\nError: Audio file not found: {audio_file}")
        print("Please provide a valid audio file path")
        return

    # Transcribe
    transcription = transcribe_audio(model, processor, audio_file)

    print("\n" + "="*60)
    print("Transcription Result:")
    print("="*60)
    print(transcription)
    print("="*60)

    # Test with multiple files if available
    print("\n\nTesting with more audio files...")
    for i in range(1, 11):
        audio_file = f"datasets/base/audio/002-04-{i:02d}.wav"
        if os.path.exists(audio_file):
            try:
                transcription = transcribe_audio(model, processor, audio_file)
                print(f"\n  Result: {transcription}")
            except Exception as e:
                print(f"\n  Error: {e}")
        else:
            break

    print("\n✓ Testing complete!")

if __name__ == "__main__":
    main()
