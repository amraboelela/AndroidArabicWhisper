#!/usr/bin/env python3
"""
Transcribe audio segments using the trained model
"""
import json
import torch
import torchaudio
import glob
import os
from improved_transformer import ImprovedDecoderTransformer

device = torch.device("cpu")

def extract_mel_features(audio_path, n_mels=800, target_fps=20):
    """Extract mel spectrogram features"""
    waveform, sample_rate = torchaudio.load(audio_path)

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
        f_max=sample_rate // 2
    )

    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    return mel_features, sample_rate


def main():
    """Transcribe all segments in segments/ directory"""

    # Paths
    datasets_dir = "datasets/base"
    vocab_path = "vocabulary.json"
    model_path = "quran_model.pt"
    output_path = "001.txt"

    # Load vocabulary
    print("Loading vocabulary...")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocabulary size: {len(vocab)}")

    # Create and load model
    print("\nLoading model...")
    model = ImprovedDecoderTransformer(
        vocab_size=len(vocab),
        d_model=800,
        n_layers=5,
        n_heads=10,
        d_ff=3200,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")

    # Get all segment files
    segment_files = sorted(glob.glob(os.path.join(datasets_dir, "001-*.wav")))
    print(f"\nFound {len(segment_files)} segments")

    # Transcribe each segment
    transcriptions = []

    for segment_file in segment_files:
        segment_name = os.path.basename(segment_file)

        # Extract audio features
        audio_features, sample_rate = extract_mel_features(segment_file)
        audio_batch = audio_features.unsqueeze(0).to(device)

        # Generate transcription
        with torch.no_grad():
            generated = model.generate(audio_batch, max_new_tokens=20, temperature=0.1)
            generated_ids = [idx for idx in generated[0].tolist() if idx not in [0, 1, 2]]
            generated_words = [vocab[idx] for idx in generated_ids]

        transcription = ' '.join(generated_words)
        transcriptions.append(transcription)

        print(f"{segment_name}: {transcription}")

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for line in transcriptions:
            f.write(line + "\n")

    print(f"\n✓ Saved transcriptions to {output_path}")
    print(f"\nFull transcription:")
    print(' '.join(transcriptions))


if __name__ == "__main__":
    main()
