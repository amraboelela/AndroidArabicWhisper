#!/usr/bin/env python3
"""
Generate augmented audio files with pitch and speed variations
Creates 8 variations for each mic quality audio file:
- pitch_minus2, pitch_minus4, pitch_plus2, pitch_plus4 (semitone shifts)
- speed_minus10, speed_minus20, speed_plus10, speed_plus20 (percentage changes)

Usage: python3 generate_augmented.py <dataset_name> <segment_name>
       python3 generate_augmented.py Quran-A 001
       python3 generate_augmented.py Quran-A 002-04
"""
import sys
import os
import glob
import numpy as np
import torchaudio
import torchaudio.functional as F
import torch

def pitch_shift(waveform, sample_rate, n_steps):
    """
    Shift pitch by n_steps semitones WITHOUT changing duration
    Uses phase vocoder (proper pitch shifting)

    Args:
        waveform: torch.Tensor [channels, samples]
        sample_rate: int
        n_steps: int (positive = higher pitch, negative = lower pitch)

    Returns:
        tuple: (shifted_waveform, sample_rate)
    """
    # Use torchaudio's pitch_shift (phase vocoder method)
    # This changes pitch WITHOUT changing duration or sample rate
    shifted = F.pitch_shift(waveform, sample_rate, n_steps)

    return shifted, sample_rate

def speed_change(waveform, sample_rate, speed_factor):
    """
    Change speed by speed_factor WITHOUT changing pitch
    Uses phase vocoder time-stretching

    Args:
        waveform: torch.Tensor [channels, samples]
        sample_rate: int
        speed_factor: float (1.0 = normal, <1.0 = slower, >1.0 = faster)

    Returns:
        tuple: (speed_changed_waveform, sample_rate)
    """
    if speed_factor == 1.0:
        return waveform, sample_rate

    # For time_stretch, rate parameter is inverse of speed_factor
    # rate > 1.0 means faster playback (shorter duration)
    # rate < 1.0 means slower playback (longer duration)
    rate = speed_factor

    # Compute spectrogram parameters
    n_freq = 1025  # Number of frequency bins (for 16kHz: n_fft=2048 -> n_freq=1025)
    hop_length = 512

    # time_stretch expects input shape [channel, freq, time]
    # We need to compute spectrogram first
    result_channels = []
    for channel in waveform:
        # Add batch dimension for spectrogram
        channel_batched = channel.unsqueeze(0)

        # Compute spectrogram
        spec = torch.stft(
            channel_batched,
            n_fft=2048,
            hop_length=hop_length,
            win_length=2048,
            window=torch.hann_window(2048),
            return_complex=True
        )

        # Apply time stretch
        stretched_spec = F.phase_vocoder(spec, rate=rate, phase_advance=hop_length)

        # Inverse STFT to get waveform back
        stretched_wave = torch.istft(
            stretched_spec,
            n_fft=2048,
            hop_length=hop_length,
            win_length=2048,
            window=torch.hann_window(2048),
            return_complex=False
        )

        result_channels.append(stretched_wave.squeeze(0))

    # Stack channels
    if len(result_channels) > 1:
        result = torch.stack(result_channels)
    else:
        result = result_channels[0].unsqueeze(0)

    return result, sample_rate

def generate_augmentations(input_file, dataset_name, segment_name, surah_num):
    """
    Generate all 8 augmented versions of an audio file
    Saves to audio/mic/augmented/{aug_type}/{surah_num}/
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(input_file)

    filename = os.path.basename(input_file)

    augmentations = {
        'pitch/minus4': lambda w, sr: pitch_shift(w, sr, -4),
        'pitch/minus2': lambda w, sr: pitch_shift(w, sr, -2),
        'pitch/plus2': lambda w, sr: pitch_shift(w, sr, 2),
        'pitch/plus4': lambda w, sr: pitch_shift(w, sr, 4),
        'speed/minus20': lambda w, sr: speed_change(w, sr, 0.80),
        'speed/minus10': lambda w, sr: speed_change(w, sr, 0.90),
        'speed/plus10': lambda w, sr: speed_change(w, sr, 1.10),
        'speed/plus20': lambda w, sr: speed_change(w, sr, 1.20),
    }

    results = []
    for aug_name, aug_func in augmentations.items():
        # Structure: audio/augmented/{aug_category}/{variation}/{surah_num}/
        output_dir = f"../datasets/{dataset_name}/audio/augmented/{aug_name}/{surah_num}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, filename)

        # Skip if already exists
        if os.path.exists(output_file):
            results.append((aug_name, 'skipped'))
            continue

        try:
            # Apply augmentation
            aug_waveform, aug_sample_rate = aug_func(waveform, sample_rate)

            # Save augmented audio
            torchaudio.save(output_file, aug_waveform, aug_sample_rate)
            results.append((aug_name, 'success'))
        except Exception as e:
            results.append((aug_name, f'error: {e}'))

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_augmented.py <dataset_name> <segment_name>")
        print("Examples:")
        print("  python3 generate_augmented.py Quran-A 001")
        print("  python3 generate_augmented.py Quran-A 002-04")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_name = sys.argv[2]

    # Extract surah number
    surah_num = segment_name.split('-')[0]

    # Determine mic audio directory
    if '-' in segment_name and len(segment_name.split('-')) > 1:
        mic_dir = f"../datasets/{dataset_name}/audio/mic/{surah_num}/{segment_name}"
    else:
        mic_dir = f"../datasets/{dataset_name}/audio/mic/{surah_num}"

    if not os.path.exists(mic_dir):
        print(f"❌ Mic audio directory not found: {mic_dir}")
        print(f"   Please run step 4 (convert to mic quality) first")
        sys.exit(1)

    # Find all mic quality audio files
    audio_files = sorted(glob.glob(os.path.join(mic_dir, f"{segment_name}-*.wav")))

    if not audio_files:
        print(f"❌ No mic audio files found in {mic_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"GENERATING AUGMENTED AUDIO")
    print(f"Dataset: {dataset_name}, Surah (part): {segment_name}")
    print(f"Found {len(audio_files)} audio files")
    print(f"{'='*60}\n")

    total_generated = 0
    total_skipped = 0
    total_errors = 0

    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        print(f"[{i}/{len(audio_files)}] Processing: {filename}")

        results = generate_augmentations(audio_file, dataset_name, segment_name, surah_num)

        for aug_name, status in results:
            if status == 'success':
                print(f"  ✓ {aug_name}")
                total_generated += 1
            elif status == 'skipped':
                print(f"  ⊘ {aug_name} (already exists)")
                total_skipped += 1
            else:
                print(f"  ✗ {aug_name}: {status}")
                total_errors += 1
        print()

    print(f"{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated: {total_generated}")
    print(f"Skipped (already exists): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"{'='*60}")
    print(f"\nAugmented audio saved to:")
    print(f"  ../datasets/{dataset_name}/audio/augmented/{{category}}/{{variation}}/{surah_num}/")
    print("Augmentation structure:")
    print("  - pitch/minus4/   - pitch/minus2/   - pitch/plus2/   - pitch/plus4/")
    print("  - speed/minus20/  - speed/minus10/  - speed/plus10/  - speed/plus20/")

if __name__ == "__main__":
    main()
