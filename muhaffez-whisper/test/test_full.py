#!/usr/bin/env python3
"""
Test encoder-decoder model on complete segments
Usage: python3 test_full.py <dataset_name> <surah_part>
Examples:
  python3 test_full.py Quran-A 001       # Test on Al-Fatiha (001)
  python3 test_full.py Quran-A 002-04    # Test on Al-Baqara part 4
"""
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import os
import sys
sys.path.append("..")
from tools.encoder_decoder_transformer import EncoderDecoderTransformer


def load_mel_features(mel_path):
    """Load precomputed mel features from .pt file"""
    if not os.path.exists(mel_path):
        raise FileNotFoundError(f"Precomputed mel features not found: {mel_path}\nPlease run precompute_mel_features.py first")

    mel_features = torch.load(mel_path, map_location='cpu', weights_only=True)
    return mel_features


def normalize_text(text):
    """Normalize Arabic text by removing diacritics and extra spacing"""
    normalized = text.replace("َ", "").replace("ً", "").replace("ُ", "").replace("ِ", "")
    normalized = normalized.replace("ّ", "").replace("ْ", "").replace("ٌ", "").replace("ٍ", "")
    return " ".join(normalized.split())


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 test_full.py <dataset_name> <surah_part>")
        print("Examples:")
        print("  python3 test_full.py Quran-A 001")
        print("  python3 test_full.py Quran-A 002-04")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2]  # e.g., "001", "002-04"

    # Device setup (silently)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Set seed for reproducible results
    torch.manual_seed(42)
    print("🎲 Random seed set to 42 for reproducibility")

    # File paths
    mels_dir = f"../datasets/{dataset_name}/mels"
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Parse surah part name
    surah_num = surah_part.split('-')[0]
    text_path = f"../datasets/{dataset_name}/text/{surah_part}.txt"

    print(f"\n{'='*60}")
    print(f"TESTING: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create reverse mapping (id -> token)
    if isinstance(vocab, dict):
        id_to_token = {v: k for k, v in vocab.items()}
    else:
        id_to_token = {i: t for i, t in enumerate(vocab)}

    # Load reference text
    if not os.path.exists(text_path):
        print(f"❌ Error: Text file not found: {text_path}")
        sys.exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        expected_texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(expected_texts)} transcriptions")

    # Load segment paths (mel files from mels directory)
    # Check if surah_part has multiple parts (e.g., "002-04")
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        # Multi-part surah (e.g., "002-04") - look in subdirectory
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
    else:
        # Single surah (e.g., "001") - look directly in surah folder
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, f"{surah_part}-*.pt")))

    if not segment_files:
        # Try the subdirectory path as fallback
        segment_files = sorted(glob.glob(os.path.join(mels_dir, surah_num, surah_part, f"{surah_part}-*.pt")))
        if not segment_files:
            print(f"❌ Error: No mel files found in {mels_dir}/{surah_num}/{surah_part}-*.pt")
            print(f"       or {mels_dir}/{surah_num}/{surah_part}/{surah_part}-*.pt")
            sys.exit(1)

    print(f"Found {len(segment_files)} audio segments")

    if len(segment_files) != len(expected_texts):
        print(f"⚠️  Warning: {len(segment_files)} segments vs {len(expected_texts)} text lines")

    # Create model
    n_mels = 40
    model = EncoderDecoderTransformer(
        vocab_size=len(id_to_token),
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1,
        n_mels=n_mels
    ).to(device)

    # Load model weights
    print(f"\nLoading trained weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✓ Model loaded successfully!")
    except RuntimeError as e:
        print(f"⚠️  Error: model shape mismatch while loading — {e}")
        sys.exit(1)

    # Run tests
    print(f"\n{'='*60}")
    print("TESTING SEGMENTS (Full)")
    print(f"{'='*60}\n")

    total_correct = 0
    total_tokens = 0

    for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
        segment_name = os.path.basename(segment_file)

        # Extract mel features
        mel_features = load_mel_features(segment_file)
        audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

        # Calculate audio duration from mel features (100 fps)
        audio_duration = mel_features.shape[0] / 100.0

        # Generate transcription
        with torch.no_grad():
            generated_ids = model.generate(
                audio_batch,
                max_new_tokens=50,
                temperature=1.0,
                min_tokens=1,
                use_sampling=False,
                audio_duration_seconds=audio_duration
            )

        # Convert to text
        tokens = generated_ids[0].tolist()
        if tokens and tokens[0] == 1:
            tokens = tokens[1:]
        if 2 in tokens:
            tokens = tokens[:tokens.index(2)]
        generated_words = [id_to_token[idx] for idx in tokens if idx in id_to_token]
        generated_text = " ".join(generated_words)

        # Check match
        normalized_generated = normalize_text(generated_text)
        normalized_expected = normalize_text(expected_text)
        match = "✓" if normalized_generated == normalized_expected else "✗"

        # Show only first 10 samples
        if i <= 10:
            print(f"[{i}/{len(segment_files)}] {segment_name}")
            print(f"Expected: {expected_text}")
            print(f"Generated: {generated_text}")
            print(f"Match: {match}\n")

        # Token-level accuracy
        expected_words = expected_text.split()
        min_len = min(len(expected_words), len(generated_words))
        total_correct += sum(1 for j in range(min_len) if generated_words[j] == expected_words[j])
        total_tokens += len(expected_words)

    # Summary
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    print(f"TEST RESULTS: {surah_part}")
    print(f"Token accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
    print(f"Segments tested: {len(segment_files)}")


if __name__ == "__main__":
    main()
