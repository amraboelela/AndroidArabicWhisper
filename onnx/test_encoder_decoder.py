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


def extract_mel_features(audio_path, n_mels=80):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Whisper parameters (100 fps: 16000 / 160 = 100)
    n_fft = 400
    hop_length = 160

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


def normalize_text(text):
    """Normalize Arabic text by removing diacritics and extra spacing"""
    # Remove common Arabic diacritics
    normalized = text.replace("َ", "").replace("ً", "").replace("ُ", "").replace("ِ", "")
    normalized = normalized.replace("ّ", "").replace("ْ", "").replace("ٌ", "").replace("ٍ", "")
    # Normalize spacing
    return " ".join(normalized.split())


def test_encoder_decoder():
    """Evaluate trained encoder-decoder model on Al-Fatiha and Al-Baqara segments"""

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

    # Set seed for reproducible results
    torch.manual_seed(42)
    print("🎲 Random seed set to 42 for reproducibility")

    # -------------------------------
    # File paths
    # -------------------------------
    datasets_dir = "datasets/base"
    vocab_path = "vocabulary.json"
    model_path = "encoder_decoder_model.pt"

    # Test datasets
    test_sets = [
        {
            "name": "Al-Fatiha (001)",
            "text_path": os.path.join(datasets_dir, "001.txt"),
            "pattern": "001-*.wav"
        },
        {
            "name": "Al-Baqara (002-01)",
            "text_path": os.path.join(datasets_dir, "002-01.txt"),
            "pattern": "002-01-*.wav"
        }
    ]

    # -------------------------------
    # Load vocabulary
    # -------------------------------
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create reverse mapping (id -> token)
    if isinstance(vocab, dict):
        # If vocab is {word: id}, reverse it
        id_to_token = {v: k for k, v in vocab.items()}
    else:
        # If vocab is a list, create index mapping
        id_to_token = {i: t for i, t in enumerate(vocab)}

    print(f"Vocabulary size: {len(id_to_token)}")

    # -------------------------------
    # Create model (128-dimension)
    # -------------------------------
    n_mels = 80
    model = EncoderDecoderTransformer(
        vocab_size=len(id_to_token),
        d_model=128,           # Smaller dimension
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,             # 128/4 = 32 dim per head
        d_ff=512,              # 4x d_model
        dropout=0.1,
        n_mels=n_mels
    ).to(device)

    # -------------------------------
    # Load model weights
    # -------------------------------
    print(f"Loading trained weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✓ Model loaded successfully!")
    except RuntimeError as e:
        print(f"⚠️  Error: model shape mismatch while loading — {e}")
        print("→ Please verify that d_model, n_heads, and layer counts match training config.")
        return

    # -------------------------------
    # Run tests on all datasets
    # -------------------------------
    overall_correct = 0
    overall_tokens = 0

    for test_set in test_sets:
        print(f"\n{'='*60}")
        print(f"Testing: {test_set['name']}")
        print(f"{'='*60}")

        # Load reference text
        with open(test_set["text_path"], "r", encoding="utf-8") as f:
            expected_texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(expected_texts)} transcriptions")

        # Load audio segments
        segment_files = sorted(glob.glob(os.path.join(datasets_dir, test_set["pattern"])))
        print(f"Found {len(segment_files)} audio segments")

        if len(segment_files) != len(expected_texts):
            print(f"⚠️  Warning: {len(segment_files)} segments vs {len(expected_texts)} text lines")

        total_correct = 0
        total_tokens = 0

        for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
            segment_name = os.path.basename(segment_file)
            print(f"\n[Segment {i}/{len(segment_files)}] {segment_name}")
            print(f"Expected: {expected_text}")

            # Extract mel features and convert to Whisper format (batch, n_mels, time)
            mel_features = extract_mel_features(segment_file)
            # mel_features is (time, n_mels), need (n_mels, time) for Whisper
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            # Calculate audio duration in seconds
            waveform, sample_rate = torchaudio.load(segment_file)
            audio_duration = waveform.shape[1] / sample_rate
            print(f"Audio duration: {audio_duration:.2f}s")

            # Generate transcription (use greedy decoding for deterministic results)
            with torch.no_grad():
                generated_ids = model.generate(
                    audio_batch,
                    max_new_tokens=50,
                    temperature=1.0,
                    min_tokens=1,
                    use_sampling=False,  # Use greedy decoding for testing
                    audio_duration_seconds=audio_duration
                )

            # Convert token IDs to words
            tokens = generated_ids[0].tolist()
            if tokens and tokens[0] == 1:
                tokens = tokens[1:]
            if 2 in tokens:
                tokens = tokens[:tokens.index(2)]
            generated_words = [id_to_token[idx] for idx in tokens if idx in id_to_token]
            generated_text = " ".join(generated_words)

            print(f"Generated: {generated_text}")

            # Token-level comparison with text normalization
            normalized_generated = normalize_text(generated_text)
            normalized_expected = normalize_text(expected_text)
            match = "✓" if normalized_generated == normalized_expected else "✗"
            print(f"Match: {match}")

            # Token-level accuracy (comparing raw tokens)
            expected_words = expected_text.split()
            min_len = min(len(expected_words), len(generated_words))
            total_correct += sum(1 for j in range(min_len) if generated_words[j] == expected_words[j])
            total_tokens += len(expected_words)

        # Dataset summary
        accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
        print(f"\n{test_set['name']} RESULTS")
        print("="*60)
        print(f"Token accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
        print(f"Segments tested: {len(segment_files)}")

        overall_correct += total_correct
        overall_tokens += total_tokens

    # -------------------------------
    # Overall Summary
    # -------------------------------
    overall_accuracy = (overall_correct / overall_tokens * 100) if overall_tokens > 0 else 0.0
    print(f"\n{'='*60}")
    print("OVERALL RESULTS (ALL DATASETS)")
    print("="*60)
    print(f"Token accuracy: {overall_correct}/{overall_tokens} ({overall_accuracy:.1f}%)")
    print(f"Total segments tested: {sum(len(glob.glob(os.path.join(datasets_dir, ts['pattern']))) for ts in test_sets)}")


if __name__ == "__main__":
    test_encoder_decoder()
