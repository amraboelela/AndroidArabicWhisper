#!/usr/bin/env python3
"""
Test encoder-decoder model using curriculum approach (progressive chunk sizes)
Usage: python3 test_curriculum.py <dataset_name> <surah_part>
Examples:
  python3 test_curriculum.py Quran-A 001       # Test on Al-Fatiha (001)
  python3 test_curriculum.py Quran-A 002-04    # Test on Al-Baqara part 4
"""
import json
import torch
import torchaudio
import glob
import os
import sys
sys.path.append("..")
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer

# Curriculum settings
CHUNK_DURATION = 1.3  # seconds per word
WORDS_PER_CHUNK = 1   # words per chunk


def extract_mel_features(audio_path, n_mels=80, target_seconds=None):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim to target seconds if specified
    if target_seconds is not None:
        num_samples = int(sample_rate * target_seconds)
        if waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]

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


def calculate_curriculum_stages(transcriptions):
    """Determine curriculum stages based on max words"""
    max_words = max(len(t.split()) for t in transcriptions)
    stages = []
    stage_num = 1

    while stage_num <= max_words:
        target_seconds = stage_num * CHUNK_DURATION
        target_words = stage_num * WORDS_PER_CHUNK
        stages.append((target_seconds, target_words, stage_num))
        stage_num += 1

    # Final stage: full
    stages.append((None, None, stage_num))
    return stages


def test_stage(model, segment_files, expected_texts, id_to_token, device,
               stage_num, target_seconds, target_words):
    """Test model on one curriculum stage"""

    audio_desc = f"{target_seconds:.1f}s" if target_seconds else "full"
    text_desc = f"{target_words} word(s)" if target_words else "full"

    print(f"\n{'='*60}")
    print(f"CURRICULUM STAGE {stage_num}: {audio_desc} → {text_desc}")
    print(f"{'='*60}\n")

    total_correct = 0
    total_tokens = 0
    testable_segments = 0

    for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
        # Check if segment has enough words for this stage
        expected_words = expected_text.split()
        if target_words and len(expected_words) < target_words:
            continue

        testable_segments += 1
        segment_name = os.path.basename(segment_file)

        # Get expected text for this stage
        if target_words:
            stage_expected = " ".join(expected_words[:target_words])
        else:
            stage_expected = expected_text

        # Extract mel features (trimmed to target_seconds)
        mel_features = extract_mel_features(segment_file, target_seconds=target_seconds)
        audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

        # Calculate audio duration
        if target_seconds:
            audio_duration = target_seconds
        else:
            waveform, sample_rate = torchaudio.load(segment_file)
            audio_duration = waveform.shape[1] / sample_rate

        # Generate transcription
        with torch.no_grad():
            max_tokens = (target_words * 10) if target_words else 50
            generated_ids = model.generate(
                audio_batch,
                max_new_tokens=max_tokens,
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

        # Show only target words if specified
        if target_words:
            display_words = generated_words[:target_words] if len(generated_words) >= target_words else generated_words
        else:
            display_words = generated_words

        generated_text = " ".join(display_words)

        # Check match
        normalized_generated = normalize_text(generated_text)
        normalized_expected = normalize_text(stage_expected)
        match = "✓" if normalized_generated == normalized_expected else "✗"

        if i <= 3 or match == "✗":  # Show first 3 and failures
            print(f"[{i}] {segment_name}")
            print(f"  Expected: {stage_expected}")
            print(f"  Generated: {generated_text}")
            print(f"  {match}\n")

        # Token-level accuracy
        stage_expected_words = stage_expected.split()
        min_len = min(len(stage_expected_words), len(display_words))
        total_correct += sum(1 for j in range(min_len) if display_words[j] == stage_expected_words[j])
        total_tokens += len(stage_expected_words)

    # Stage summary
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    print(f"Stage {stage_num} Results:")
    print(f"  Token accuracy: {total_correct}/{total_tokens} ({accuracy:.1f}%)")
    print(f"  Testable segments: {testable_segments}/{len(segment_files)}")

    return total_correct, total_tokens


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 test_curriculum.py <dataset_name> <surah_part>")
        print("Examples:")
        print("  python3 test_curriculum.py Quran-A 001")
        print("  python3 test_curriculum.py Quran-A 002-04")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2]  # e.g., "001", "002-04"

    # Device setup
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

    # File paths
    datasets_dir = f"../datasets/{dataset_name}/audio"
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Parse surah part name
    surah_num = surah_part.split('-')[0]
    text_path = f"../datasets/{dataset_name}/text/{surah_part}.txt"

    print(f"\n{'='*60}")
    print(f"CURRICULUM TESTING: {surah_part}")
    print(f"Dataset: {dataset_name}")
    print(f"Chunk size: {CHUNK_DURATION}s → {WORDS_PER_CHUNK} word(s)")
    print(f"{'='*60}\n")

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create reverse mapping
    if isinstance(vocab, dict):
        id_to_token = {v: k for k, v in vocab.items()}
    else:
        id_to_token = {i: t for i, t in enumerate(vocab)}
    print(f"Vocabulary size: {len(id_to_token)}")

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

    # Calculate curriculum stages
    stages = calculate_curriculum_stages(expected_texts)
    print(f"\nCurriculum has {len(stages)} stages")

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

    # Test through all stages
    overall_correct = 0
    overall_tokens = 0

    for target_seconds, target_words, stage_num in stages:
        stage_correct, stage_tokens = test_stage(
            model,
            segment_files,
            expected_texts,
            id_to_token,
            device,
            stage_num,
            target_seconds,
            target_words
        )
        overall_correct += stage_correct
        overall_tokens += stage_tokens

    # Overall summary
    overall_accuracy = (overall_correct / overall_tokens * 100) if overall_tokens > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"OVERALL CURRICULUM TEST RESULTS: {surah_part}")
    print(f"{'='*60}")
    print(f"Token accuracy: {overall_correct}/{overall_tokens} ({overall_accuracy:.1f}%)")
    print(f"Total stages: {len(stages)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
