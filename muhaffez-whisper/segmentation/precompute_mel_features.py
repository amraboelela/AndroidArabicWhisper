#!/usr/bin/env python3
"""
Precompute mel spectrograms for all audio files in all datasets
Saves them as .pt files alongside the audio files for fast loading during training
Usage: python3 precompute_mel_features.py [dataset_name]
Example:
  python3 precompute_mel_features.py          # Process all datasets
  python3 precompute_mel_features.py Quran-A  # Process only Quran-A
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchaudio
import glob
import os
from pathlib import Path

def extract_mel_features(audio_path, n_mels=80):
    """Extract mel features from audio (same as training scripts)"""
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
        f_max=sample_rate // 2
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    mel_features = mel_spec.squeeze(0).transpose(0, 1)

    # Global Whisper normalization
    mel_mean = -4.2677
    mel_std = 4.5689
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features

def precompute_dataset(dataset_path):
    """Precompute mel features for all audio files in a dataset"""
    dataset_name = os.path.basename(dataset_path)
    print(f"\n{'='*60}")
    print(f"PRECOMPUTING MEL FEATURES - DATASET: {dataset_name}")
    print(f"{'='*60}\n")

    # Find all audio files
    audio_dir = f"{dataset_path}/audio"
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        return

    audio_files = sorted(glob.glob(f"{audio_dir}/**/*.wav", recursive=True))
    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files")

    processed = 0
    skipped = 0
    errors = 0

    for audio_file in audio_files:
        # Create mel feature path (same directory, .pt extension)
        mel_path = audio_file.replace('/audio/', '/mels/').replace('.wav', '.pt')

        # Skip if already exists
        if os.path.exists(mel_path):
            skipped += 1
            continue

        try:
            # Extract and save mel features
            mel_features = extract_mel_features(audio_file)
            torch.save(mel_features, mel_path)
            processed += 1

            if processed % 50 == 0:
                print(f"  Processed {processed}/{len(audio_files) - skipped} files...", flush=True)

        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            errors += 1

    print(f"\n✓ Precomputation complete!")
    print(f"  Processed: {processed} files")
    if skipped > 0:
        print(f"  Skipped (already exists): {skipped} files")
    if errors > 0:
        print(f"  Errors: {errors} files")

def main():
    if len(sys.argv) > 1:
        # Process specific dataset
        dataset_name = sys.argv[1]
        dataset_path = f"../datasets/{dataset_name}"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            sys.exit(1)
        precompute_dataset(dataset_path)
    else:
        # Process all datasets
        datasets_dir = "../datasets"
        if not os.path.exists(datasets_dir):
            print(f"❌ Datasets directory not found: {datasets_dir}")
            sys.exit(1)

        datasets = [d for d in os.listdir(datasets_dir)
                   if os.path.isdir(os.path.join(datasets_dir, d)) and not d.startswith('.')]

        if not datasets:
            print(f"❌ No datasets found in {datasets_dir}")
            sys.exit(1)

        print(f"Found {len(datasets)} dataset(s): {datasets}")

        for dataset_name in sorted(datasets):
            dataset_path = os.path.join(datasets_dir, dataset_name)
            precompute_dataset(dataset_path)

    print("\n✓ All datasets processed successfully!")

if __name__ == "__main__":
    main()
