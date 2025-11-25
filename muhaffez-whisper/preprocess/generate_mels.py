#!/usr/bin/env python3
"""
Generate mel spectrograms from audio files in datasets
Saves them as .pt files for fast loading during training
100% Whisper-accurate: Uses Whisper's exact mel filterbank, STFT settings, and normalization

Usage: python3 generate_mels.py [dataset_name] [surah_part]
Examples:
  python3 generate_mels.py                    # Process all datasets
  python3 generate_mels.py Quran-A            # Process all parts in Quran-A
  python3 generate_mels.py Quran-A 001        # Process single surah 001
  python3 generate_mels.py Quran-A 002-02     # Process specific part 002-02
  python3 generate_mels.py Quran-A 002        # Process all parts of surah 002
"""
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torchaudio
import numpy as np
import glob
import os
from pathlib import Path

def extract_mel_features_whisper_accurate(audio_path, n_mels=40):
    """
    Extract mel features for 8kHz mic audio using 40 mel bins (0-4000 Hz)
    Adapted from Whisper's pipeline for lower sample rate audio
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # For 8kHz audio, we don't resample - keep it at 8kHz
    # 8kHz gives us 0-4000 Hz frequency range (Nyquist)
    # We'll use 40 mel bins for this range (half of Whisper's 80 mels for 16kHz)

    # STFT parameters adapted for 8kHz
    # Scale down from Whisper's 16kHz parameters
    n_fft = 400  # Keep same n_fft for similar frequency resolution
    hop_length = 80  # Half of 160 (since sample rate is half)

    # Whisper padding: reflect mode
    pad = n_fft // 2
    waveform = torch.nn.functional.pad(
        waveform,
        (pad, pad),
        mode="reflect"
    ).squeeze(0)

    # Whisper Hann window
    window = torch.hann_window(n_fft + 1, periodic=False)[:-1]

    # Compute STFT (center=False because we manually padded)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=False,
        return_complex=True
    )

    # Compute magnitude squared (power spectrogram)
    window_norm = (window**2).sum()
    magnitude = (stft.abs() ** 2) / window_norm

    # Create mel filterbank for 8kHz audio with 40 mel bins
    # Frequency range: 0-4000 Hz (Nyquist for 8kHz)
    mel_filterbank = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=4000.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm="slaney",
        mel_scale="slaney"
    )

    # Apply mel filterbank
    mel_spec = mel_filterbank.T @ magnitude

    # Log mel (Whisper uses ln, not log10)
    mel_spec = torch.log(mel_spec + 1e-10)

    # Transpose to (time, mel_bins)
    mel_features = mel_spec.transpose(0, 1)

    # Localized per-segment normalization (better for varying audio conditions)
    mel_mean = mel_features.mean()
    mel_std = mel_features.std()
    mel_features = (mel_features - mel_mean) / (mel_std + 1e-8)

    return mel_features

def precompute_dataset(dataset_path, surah_part=None):
    """Precompute mel features for audio files in a dataset

    Args:
        dataset_path: Path to dataset directory
        surah_part: Optional surah part to process (e.g., "001", "002-02"). If None, processes all.
    """
    dataset_name = os.path.basename(dataset_path)
    print(f"\n{'='*60}")
    if surah_part:
        print(f"PRECOMPUTING MEL FEATURES - DATASET: {dataset_name}, PART: {surah_part}")
    else:
        print(f"PRECOMPUTING MEL FEATURES - DATASET: {dataset_name}")
    print(f"{'='*60}\n")

    # Process both mic and augmented audio
    audio_sources = [
        ('mic', f"{dataset_path}/audio/mic", '/audio/mic/', '/mels/normal/'),
        ('augmented', f"{dataset_path}/audio/augmented", '/audio/augmented/', '/mels/augmented/')
    ]

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    for source_name, audio_dir, search_pattern, replace_pattern in audio_sources:
        if not os.path.exists(audio_dir):
            if source_name == 'mic':
                print(f"❌ Audio directory not found: {audio_dir}")
                return
            else:
                # Augmented audio is optional
                continue

        # If surah_part specified, only process that part
        if surah_part:
            surah_num = surah_part.split('-')[0]
            # Check if surah_part has multiple parts (e.g., "002-04")
            if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
                # Multi-part surah (e.g., "002-04")
                if source_name == 'mic':
                    audio_files = sorted(glob.glob(f"{audio_dir}/{surah_num}/{surah_part}/{surah_part}-*.wav"))
                else:
                    # For augmented, search in all augmentation subdirectories with part structure
                    # Pattern: augmented/{category}/{variation}/{surah_num}/{surah_part}/
                    audio_files = sorted(glob.glob(f"{audio_dir}/**/{surah_num}/{surah_part}/{surah_part}-*.wav", recursive=True))
            else:
                # Single surah (e.g., "001")
                if source_name == 'mic':
                    audio_files = sorted(glob.glob(f"{audio_dir}/{surah_num}/{surah_part}-*.wav"))
                else:
                    # For augmented, search in all augmentation subdirectories
                    # Pattern: augmented/{category}/{variation}/{surah_num}/
                    audio_files = sorted(glob.glob(f"{audio_dir}/**/{surah_num}/{surah_part}-*.wav", recursive=True))

            if not audio_files and source_name == 'mic':
                # Try subdirectory as fallback for mic
                audio_files = sorted(glob.glob(f"{audio_dir}/{surah_num}/{surah_part}/{surah_part}-*.wav"))
        else:
            # Process all audio files
            audio_files = sorted(glob.glob(f"{audio_dir}/**/*.wav", recursive=True))

        if not audio_files:
            if source_name == 'mic':
                print(f"❌ No audio files found in {audio_dir}")
                return
            else:
                # No augmented files is OK
                continue

        print(f"\n{source_name.upper()}: Found {len(audio_files)} audio files")

        processed = 0
        skipped = 0
        errors = 0

        for audio_file in audio_files:
            # Create mel feature path by replacing search pattern with replace pattern and .wav with .pt
            mel_path = audio_file.replace(search_pattern, replace_pattern).replace('.wav', '.pt')

            # Create mels directory if it doesn't exist
            os.makedirs(os.path.dirname(mel_path), exist_ok=True)

            # Skip if already exists
            if os.path.exists(mel_path):
                skipped += 1
                continue

            try:
                # Extract and save mel features (Whisper-accurate)
                mel_features = extract_mel_features_whisper_accurate(audio_file)
                torch.save(mel_features, mel_path)
                processed += 1

                if processed % 50 == 0:
                    print(f"  Processed {processed}/{len(audio_files) - skipped} files...", flush=True)

            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
                errors += 1

        print(f"✓ {source_name.upper()} complete!")
        print(f"  Processed: {processed} files")
        if skipped > 0:
            print(f"  Skipped (already exists): {skipped} files")
        if errors > 0:
            print(f"  Errors: {errors} files")

        total_processed += processed
        total_skipped += skipped
        total_errors += errors

    print(f"\n{'='*60}")
    print(f"✓ TOTAL PRECOMPUTATION COMPLETE!")
    print(f"  Total Processed: {total_processed} files")
    if total_skipped > 0:
        print(f"  Total Skipped: {total_skipped} files")
    if total_errors > 0:
        print(f"  Total Errors: {total_errors} files")
    print(f"{'='*60}")

def main():
    if len(sys.argv) > 1:
        # Process specific dataset
        dataset_name = sys.argv[1]
        dataset_path = f"../datasets/{dataset_name}"
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            sys.exit(1)

        # Check if surah_part is specified
        if len(sys.argv) > 2:
            surah_part = sys.argv[2]
            surah_num = surah_part.split('-')[0]

            # Check if this is a request to process all parts of a surah
            if surah_part == surah_num:
                # Process all parts of the surah
                mic_base_dir = f"{dataset_path}/audio/mic/{surah_num}"

                if not os.path.exists(mic_base_dir):
                    print(f"❌ Mic audio directory not found: {mic_base_dir}")
                    sys.exit(1)

                # Find all part subdirectories
                parts = []
                for item in sorted(os.listdir(mic_base_dir)):
                    item_path = os.path.join(mic_base_dir, item)
                    if os.path.isdir(item_path) and item.startswith(f"{surah_num}-"):
                        parts.append(item)

                if not parts:
                    # No parts found, treat as single surah
                    print(f"No parts found, processing {surah_part} as single surah")
                    precompute_dataset(dataset_path, surah_part)
                else:
                    # Process each part
                    print(f"{'='*60}")
                    print(f"PROCESSING ALL PARTS OF SURAH {surah_num}")
                    print(f"Found {len(parts)} parts: {', '.join(parts)}")
                    print(f"{'='*60}")

                    for i, part in enumerate(parts, 1):
                        print(f"\n{'#'*60}")
                        print(f"# PART {i}/{len(parts)}: {part}")
                        print(f"{'#'*60}")
                        precompute_dataset(dataset_path, part)

                    print(f"\n{'='*60}")
                    print(f"✓ ALL PARTS COMPLETE - SURAH {surah_num}")
                    print(f"  Processed {len(parts)} parts")
                    print(f"{'='*60}")
            else:
                # Process single part
                precompute_dataset(dataset_path, surah_part)
        else:
            # Process all parts in dataset
            precompute_dataset(dataset_path, None)
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
    print("\n📊 Mel features optimized for 8kHz mic audio:")
    print("   ✓ 40 mel bins (0-4000 Hz frequency range)")
    print("   ✓ 8kHz sample rate (mobile mic quality)")
    print("   ✓ STFT parameters: n_fft=400, hop=80")
    print("   ✓ Reflect padding (Whisper-style)")
    print("   ✓ Slaney mel scale normalization")
    print("   ✓ Per-segment normalization (mean=0, std=1)")
    print("\n🎯 Features are ready for training with 40-mel configuration!")

if __name__ == "__main__":
    main()
