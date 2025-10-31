#!/usr/bin/env python3
"""
Test encoder-decoder model on Al-Fatiha segments
"""
import json
import torch
import torchaudio
import glob
import os
from encoder_decoder_transformer import EncoderDecoderTransformer


def extract_mel_features(audio_path, n_mels=128, target_fps=20):
    """Extract normalized mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    hop_length = sample_rate // target_fps
    n_fft = 2048

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2,
    )

    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    # Normalize (same as training)
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-5)

    return mel_features


def test_encoder_decoder():
    """Evaluate trained encoder-decoder model on Al-Fatiha segments"""

    # -------------------------------
    # Device setup
    # -------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Metal GPU (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️  Using CPU (slower)")

    print(f"Device: {device}")

    # -------------------------------
    # File paths
    # -------------------------------
    segments_dir = "segments"
    text_path = os.path.join(segments_dir, "001.txt")
    vocab_path = "vocabulary.json"
    model_path = "encoder_decoder_model.pt"

    print("\n" + "=" * 60)
    print("Testing Encoder-Decoder Model on Al-Fatiha Segments")
    print("=" * 60)

    # -------------------------------
    # Load vocabulary
    # -------------------------------
    print("\n1. Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"   Vocabulary size: {len(vocab)}")

    # -------------------------------
    # Load reference text
    # -------------------------------
    print(f"\n2. Loading expected transcriptions from {text_path}...")
    with open(text_path, "r", encoding="utf-8") as f:
        expected_texts = [line.strip() for line in f if line.strip()]
    print(f"   Loaded {len(expected_texts)} transcriptions")

    # -------------------------------
    # Load audio segments
    # -------------------------------
    segment_files = sorted(glob.glob(os.path.join(segments_dir, "001-*.wav")))
    print(f"\n3. Found {len(segment_files)} audio segments")

    if len(segment_files) != len(expected_texts):
        print(f"⚠️  Warning: {len(segment_files)} segments vs {len(expected_texts)} text lines")

    # -------------------------------
    # Create model
    # -------------------------------
    print("\n4. Creating encoder-decoder model...")
    model = EncoderDecoderTransformer(
        vocab_size=len(vocab),
        d_model=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        d_ff=256,
        dropout=0.2,
    ).to(device)

    # -------------------------------
    # Load model weights
    # -------------------------------
    print(f"\n5. Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("   ✓ Model loaded successfully!")

    # -------------------------------
    # Run tests
    # -------------------------------
    print("\n" + "=" * 60)
    print("SEGMENT-BY-SEGMENT TESTING")
    print("=" * 60)

    total_correct = 0
    total_tokens = 0

    for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
        segment_name = os.path.basename(segment_file)
        print(f"\n[Segment {i}/{len(segment_files)}] {segment_name}")
        print(f"Expected: {expected_text}")

        # Extract mel features
        audio_features = extract_mel_features(segment_file)
        audio_batch = audio_features.unsqueeze(0).to(device)

        # Generate transcription with sampling
        with torch.no_grad():
            generated_ids = model.generate(
                audio_batch,
                max_new_tokens=20,
                temperature=1.0,
                min_tokens=1,
                use_sampling=True,  # Use sampling for more diversity
            )

        # Convert token IDs to words
        generated_tokens = generated_ids[0].tolist()

        # Remove <s> at start if present
        if generated_tokens and generated_tokens[0] == 1:
            generated_tokens = generated_tokens[1:]
        # Truncate at </s> if present
        if 2 in generated_tokens:
            generated_tokens = generated_tokens[: generated_tokens.index(2)]

        # Convert to text
        generated_words = [vocab[idx] for idx in generated_tokens if idx < len(vocab)]
        generated_text = " ".join(generated_words)

        print(f"Generated: {generated_text}")

        # Compare
        expected_words = expected_text.split()
        match = "✓" if generated_text == expected_text else "✗"
        print(f"Match: {match}")

        # Token-level accuracy
        min_len = min(len(expected_words), len(generated_words))
        correct = sum(
            1 for j in range(min_len)
            if generated_words[j] == expected_words[j]
        )
        total_correct += correct
        total_tokens += len(expected_words)

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    print(f"Token accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
    print(f"Segments tested: {len(segment_files)}")


if __name__ == "__main__":
    test_encoder_decoder()
