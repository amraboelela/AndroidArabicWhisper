#!/usr/bin/env python3
"""
Test encoder-decoder model on first 3 seconds of Al-Baqara segments, expecting first 2 words
"""
import json
import torch
import torchaudio
import glob
import os
import sys
sys.path.append("../..")
from encoder_decoder_transformer import EncoderDecoderTransformer


def extract_first_seconds_mel(audio_path, n_mels=80, target_seconds=3.0):
    """Extract mel features from only the first N seconds of the audio"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Trim to first N seconds
    num_samples = int(sample_rate * target_seconds)
    if waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]

    # Whisper-like parameters (hop_length=160 → 100 fps)
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

    # Normalize like during training
    # Global Whisper normalization (more robust than per-sample)

    WHISPER_MEL_MEAN = -4.2677393

    WHISPER_MEL_STD = 4.5689974

    mel_features = (mel_features - WHISPER_MEL_MEAN) / WHISPER_MEL_STD
    return mel_features


def normalize_text(text):
    """Normalize Arabic text by removing diacritics and extra spacing"""
    normalized = text.replace("َ", "").replace("ً", "").replace("ُ", "").replace("ِ", "")
    normalized = normalized.replace("ّ", "").replace("ْ", "").replace("ٌ", "").replace("ٍ", "")
    return " ".join(normalized.split())


def test_baqara_first_3_seconds():
    """Evaluate trained model using only first 3 seconds of Al-Baqara segments"""
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Metal GPU (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")
    print(f"Device: {device}")

    torch.manual_seed(42)
    print("🎲 Random seed set to 42 for reproducibility")

    import sys
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "base"
    datasets_dir = f"../{dataset_name}/audio"
    vocab_path = "../../vocabulary.json"
    model_path = "../../models/encoder_decoder_model.pt"

    test_sets = [
        {
            "name": "Al-Baqara Part 1 (002-01)",
            "text_path": f"../{dataset_name}/text/002-01.txt",
            "pattern": "002-01-*.wav"
        },
        {
            "name": "Al-Baqara Part 2 (002-02)",
            "text_path": f"../{dataset_name}/text/002-02.txt",
            "pattern": "002-02-*.wav"
        },
        {
            "name": "Al-Baqara Part 3 (002-03)",
            "text_path": f"../{dataset_name}/text/002-03.txt",
            "pattern": "002-03-*.wav"
        }
    ]

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()} if isinstance(vocab, dict) else {i: t for i, t in enumerate(vocab)}
    print(f"Vocabulary size: {len(id_to_token)}")

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

    print(f"Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded successfully!")

    overall_correct = 0
    overall_total = 0

    for test_set in test_sets:
        print(f"\n{'='*60}")
        print(f"Testing (first 3s → first 2 words): {test_set['name']}")
        print(f"{'='*60}")

        with open(test_set["text_path"], "r", encoding="utf-8") as f:
            expected_texts = [line.strip() for line in f if line.strip()]

        segment_files = sorted(glob.glob(os.path.join(datasets_dir, test_set["pattern"])))
        print(f"Found {len(segment_files)} audio segments")

        total_correct = 0
        total_segments = 0

        for i, (segment_file, expected_text) in enumerate(zip(segment_files, expected_texts), 1):
            segment_name = os.path.basename(segment_file)
            words = expected_text.split()
            first_two_words = " ".join(words[:2]) if len(words) >= 2 else expected_text
            print(f"\n[{i:02d}/{len(segment_files)}] {segment_name}")
            print(f"Expected (first 2 words): {first_two_words}")

            # Extract first 3 seconds mel
            mel_features = extract_first_seconds_mel(segment_file, target_seconds=3.0)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    audio_batch,
                    max_new_tokens=30,
                    temperature=1.0,
                    min_tokens=1,
                    use_sampling=False,
                    audio_duration_seconds=3.0
                )

            tokens = generated_ids[0].tolist()
            if tokens and tokens[0] == 1:
                tokens = tokens[1:]
            if 2 in tokens:
                tokens = tokens[:tokens.index(2)]
            generated_words = [id_to_token[idx] for idx in tokens if idx in id_to_token]

            # Take first 2 words from model output
            generated_first_two = " ".join(generated_words[:2]) if len(generated_words) >= 2 else " ".join(generated_words)
            print(f"Generated (first 2 words): {generated_first_two}")

            # Compare normalized
            normalized_generated = normalize_text(generated_first_two)
            normalized_expected = normalize_text(first_two_words)
            match = "✓" if normalized_generated == normalized_expected else "✗"
            print(f"Match: {match}")

            if normalized_generated == normalized_expected:
                total_correct += 1
            total_segments += 1

        accuracy = (total_correct / total_segments * 100) if total_segments > 0 else 0.0
        print(f"\n{test_set['name']} (3s → first 2 words) RESULTS")
        print("="*60)
        print(f"Accuracy: {total_correct}/{total_segments} ({accuracy:.1f}%)")
        overall_correct += total_correct
        overall_total += total_segments

    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
    print(f"\n{'='*60}")
    print("OVERALL RESULTS (Al-Baqara 3s → first 2 words)")
    print("="*60)
    print(f"Accuracy: {overall_correct}/{overall_total} ({overall_accuracy:.1f}%)")


if __name__ == "__main__":
    test_baqara_first_3_seconds()
