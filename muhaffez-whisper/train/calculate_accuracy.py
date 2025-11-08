#!/usr/bin/env python3
"""
Calculate overall accuracy across all existing segments silently
Usage: python3 calculate_accuracy.py <dataset_name>
Output: Single percentage like "55%"
"""
import json
import torch
import warnings
import sys
import glob
import os

# Suppress all warnings
warnings.filterwarnings("ignore")
sys.path.append("..")
from custom_scripts.encoder_decoder_transformer import EncoderDecoderTransformer
import torchaudio


def extract_mel_features(audio_path, n_mels=80):
    """Extract Whisper-compatible mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

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
    """Normalize Arabic text"""
    normalized = text.replace("َ", "").replace("ً", "").replace("ُ", "").replace("ِ", "")
    normalized = normalized.replace("ّ", "").replace("ْ", "").replace("ٌ", "").replace("ٍ", "")
    return " ".join(normalized.split())


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 calculate_accuracy.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(42)

    # Paths
    vocab_path = "../models/vocabulary.json"
    model_path = "../models/muhaffez_whisper.pt"

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocabulary = json.load(f)

    vocab_size = len(vocabulary)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # Load model
    model = EncoderDecoderTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_heads=4,
        d_ff=512,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Find all text files
    text_files = glob.glob(f"../datasets/{dataset_name}/text/*.txt")

    total_correct = 0
    total_tokens = 0

    # Process each surah part
    for text_path in text_files:
        surah_part = os.path.splitext(os.path.basename(text_path))[0]
        surah_num = surah_part.split('-')[0]
        audio_dir = f"../datasets/{dataset_name}/audio/{surah_num}"

        # Load transcriptions
        with open(text_path, "r", encoding="utf-8") as f:
            transcriptions = [line.strip() for line in f if line.strip()]

        # Find audio segments
        segment_files = sorted(glob.glob(f"{audio_dir}/{surah_part}-*.wav"))

        if len(transcriptions) != len(segment_files):
            continue

        # Process each segment
        for seg_file, ground_truth in zip(segment_files, transcriptions):
            mel_features = extract_mel_features(seg_file)
            audio_batch = mel_features.transpose(0, 1).unsqueeze(0).to(device)

            waveform, sample_rate = torchaudio.load(seg_file)
            audio_duration = waveform.shape[1] / sample_rate

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

            predicted_words = [vocabulary[idx] for idx in tokens if idx < vocab_size]
            predicted_text = " ".join(predicted_words)

            # Normalize and compare
            pred_normalized = normalize_text(predicted_text)
            gt_normalized = normalize_text(ground_truth)

            pred_words = pred_normalized.split()
            gt_words = gt_normalized.split()

            # Count correct tokens
            for i in range(min(len(pred_words), len(gt_words))):
                if pred_words[i] == gt_words[i]:
                    total_correct += 1

            total_tokens += len(gt_words)

    # Calculate and output single percentage
    accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0.0
    print(f"{accuracy:.0f}%")


if __name__ == "__main__":
    main()
