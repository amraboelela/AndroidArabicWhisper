#!/usr/bin/env python3
import json
import torch
import torchaudio
import glob
import os
from improved_transformer import ImprovedDecoderTransformer

def extract_mel_features(audio_path, n_mels=800, target_fps=20):
    """Extract mel spectrogram features from audio"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Calculate hop length for target fps
    hop_length = sample_rate // target_fps
    n_fft = 2048

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2
    )

    # Extract mel spectrogram
    mel_spec = mel_transform(waveform)  # (1, n_mels, time)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    # Transpose to (time, n_mels)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    return mel_features


def test_model_on_segments():
    """Test the trained model on Al-Fatiha segments"""

    # Paths
    segments_dir = "segments"
    text_path = "segments/001.txt"
    vocab_path = "vocabulary.json"
    model_path = "quran_model.pt"

    print("="*60)
    print("Testing Trained Model on Al-Fatiha Segments")
    print("="*60)

    # Load vocabulary
    print("\n1. Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"   Vocabulary size: {len(vocab)}")

    # Load expected transcriptions
    print(f"\n2. Loading expected transcriptions from {text_path}...")
    with open(text_path, "r", encoding="utf-8") as f:
        expected_texts = [line.strip() for line in f if line.strip()]
    print(f"   Loaded {len(expected_texts)} transcriptions")

    # Get segment files
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "001-*.wav")))
    print(f"\n3. Found {len(segment_files)} audio segments")

    # Create model
    print("\n4. Creating model architecture...")
    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )

    # Load trained weights
    print(f"\n5. Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("   Model loaded successfully!")

    # Test each segment
    print("\n" + "="*60)
    print("SEGMENT-BY-SEGMENT TESTING")
    print("="*60)

    total_correct = 0
    total_tokens = 0

    for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
        segment_name = os.path.basename(segment_file)

        print(f"\n[Segment {i}/{len(segment_files)}] {segment_name}")
        print(f"Expected: {expected_text}")

        # Extract audio features
        audio_features = extract_mel_features(segment_file)
        audio_batch = audio_features.unsqueeze(0)

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                audio_batch,
                max_new_tokens=20,
                temperature=0.1
            )

        # Convert to words
        generated_tokens = generated_ids[0].tolist()
        # Skip <s> token at beginning if present
        if generated_tokens[0] == 1:
            generated_tokens = generated_tokens[1:]
        # Stop at </s> token if present
        if 2 in generated_tokens:
            end_idx = generated_tokens.index(2)
            generated_tokens = generated_tokens[:end_idx]

        generated_words = [vocab[idx] for idx in generated_tokens]
        generated_text = " ".join(generated_words)

        print(f"Generated: {generated_text}")

        # Compare
        expected_words = expected_text.split()
        match = "✓" if generated_text == expected_text else "✗"
        print(f"Match: {match}")

        # Calculate token accuracy
        min_len = min(len(expected_words), len(generated_words))
        correct = sum(1 for j in range(min_len) if j < len(generated_words) and generated_words[j] == expected_words[j])
        total_correct += correct
        total_tokens += len(expected_words)

    # Summary
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
    print(f"Token accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
    print(f"Segments tested: {len(segment_files)}")


if __name__ == "__main__":
    test_model_on_segments()
