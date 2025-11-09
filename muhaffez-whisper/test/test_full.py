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
# Suppress all torchaudio warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*torchaudio.*")
import torchaudio
import glob
import os
import sys
sys.path.append("..")
from tools.encoder_decoder_transformer import EncoderDecoderTransformer


def extract_mel_features(audio_path, n_mels=80):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz (Whisper standard)
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

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

    # Global Whisper normalization
    WHISPER_MEL_MEAN = -4.2677393
    WHISPER_MEL_STD = 4.5689974
    mel_features = (mel_features - WHISPER_MEL_MEAN) / WHISPER_MEL_STD

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
    datasets_dir = f"../datasets/{dataset_name}/audio"
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

    # Load audio segments
    segment_files = sorted(glob.glob(os.path.join(datasets_dir, surah_num, f"{surah_part}-*.wav")))

    if not segment_files:
        print(f"❌ Error: No audio segments found in {datasets_dir}/{surah_num}/{surah_part}-*.wav")
        sys.exit(1)

    print(f"Found {len(segment_files)} audio segments")

    if len(segment_files) != len(expected_texts):
        print(f"⚠️  Warning: {len(segment_files)} segments vs {len(expected_texts)} text lines")

    # Create model
    n_mels = 80
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
        mel_features = extract_mel_features(segment_file)
        audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

        # Calculate audio duration
        waveform, sample_rate = torchaudio.load(segment_file)
        audio_duration = waveform.shape[1] / sample_rate

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
