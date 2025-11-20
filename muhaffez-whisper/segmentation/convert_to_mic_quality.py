#!/usr/bin/env python3
"""
Convert audio segments from raw to mobile microphone quality
Usage: python3 convert_to_mic_quality.py <dataset_name> [surah_part]

Examples:
  python3 convert_to_mic_quality.py Quran-A         # Convert all parts
  python3 convert_to_mic_quality.py Quran-A 001     # Convert only 001
  python3 convert_to_mic_quality.py Quran-A 002-02  # Convert only 002-02

Converts audio from 16kHz (raw quality) to 8kHz (mobile mic quality)
"""
import sys
import os
import glob
import torchaudio

def convert_to_mic_quality(input_file, output_file, target_sample_rate=8000):
    """
    Convert audio file to mobile microphone quality (8kHz)

    Args:
        input_file: Path to input .wav file
        output_file: Path to output .wav file
        target_sample_rate: Target sampling rate (default: 8000 Hz)
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(input_file)

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target rate (8kHz for mobile mic quality)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # Save converted audio
    torchaudio.save(output_file, waveform, target_sample_rate)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_to_mic_quality.py <dataset_name> [surah_part]")
        print("Examples:")
        print("  python3 convert_to_mic_quality.py Quran-A         # Convert all parts")
        print("  python3 convert_to_mic_quality.py Quran-A 001     # Convert only 001")
        print("  python3 convert_to_mic_quality.py Quran-A 002-02  # Convert only 002-02")
        sys.exit(1)

    dataset_name = sys.argv[1]  # e.g., "Quran-A"
    surah_part = sys.argv[2] if len(sys.argv) > 2 else None  # Optional

    # If no surah_part specified, find all parts in raw directory
    if not surah_part:
        print(f"\n{'='*60}")
        print(f"CONVERTING ALL PARTS TO MOBILE MIC QUALITY")
        print(f"Dataset: {dataset_name}")
        print(f"Target sampling rate: 8kHz")
        print(f"{'='*60}\n")

        raw_audio_dir = f"../datasets/{dataset_name}/audio/raw"
        if not os.path.exists(raw_audio_dir):
            print(f"❌ Error: Raw audio directory not found: {raw_audio_dir}")
            sys.exit(1)

        # Find all .wav files recursively
        all_wav_files = sorted(glob.glob(f"{raw_audio_dir}/**/*.wav", recursive=True))
        if not all_wav_files:
            print(f"❌ Error: No audio files found in {raw_audio_dir}")
            sys.exit(1)

        # Extract unique surah parts from filenames
        surah_parts = set()
        for wav_file in all_wav_files:
            basename = os.path.basename(wav_file)
            # Extract part like "001-01" or "001" from "001-01.wav" or "002-02-05.wav"
            if basename.endswith('.wav'):
                # Match patterns like 001-01.wav or 002-02-05.wav
                parts = basename.replace('.wav', '').split('-')
                if len(parts) >= 2 and parts[1].isdigit():
                    # Multi-part like 002-02
                    surah_part_name = f"{parts[0]}-{parts[1]}"
                else:
                    # Single part like 001
                    surah_part_name = parts[0]
                surah_parts.add(surah_part_name)

        surah_parts = sorted(surah_parts)
        print(f"Found {len(surah_parts)} parts to convert: {', '.join(surah_parts)}\n")

        total_converted = 0
        total_skipped = 0
        total_errors = 0

        for part in surah_parts:
            print(f"{'='*60}")
            print(f"Converting part: {part}")
            print(f"{'='*60}")
            converted, skipped, errors = convert_part(dataset_name, part)
            total_converted += converted
            total_skipped += skipped
            total_errors += errors
            print()

        # Summary
        print(f"\n{'='*60}")
        print(f"ALL PARTS CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Total converted: {total_converted}")
        print(f"Total skipped (already exists): {total_skipped}")
        print(f"Total errors: {total_errors}")
        print(f"{'='*60}\n")
    else:
        # Convert single specified part
        print(f"\n{'='*60}")
        print(f"CONVERTING TO MOBILE MIC QUALITY: {surah_part}")
        print(f"Dataset: {dataset_name}")
        print(f"Target sampling rate: 8kHz")
        print(f"{'='*60}\n")

        converted, skipped, errors = convert_part(dataset_name, surah_part)

        # Summary
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE")
        print(f"{'='*60}")
        print(f"Converted: {converted}")
        print(f"Skipped (already exists): {skipped}")
        print(f"Errors: {errors}")
        print(f"{'='*60}\n")

def convert_part(dataset_name, surah_part):
    """Convert a specific surah part to mic quality

    Returns:
        (converted, skipped, errors): Tuple of counts
    """
    # Extract surah number
    surah_num = surah_part.split('-')[0]

    # Determine input directory (raw audio)
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        # Multi-part surah (e.g., "002-02")
        input_dir = f"../datasets/{dataset_name}/audio/raw/{surah_num}/{surah_part}"
    else:
        # Single surah (e.g., "001")
        input_dir = f"../datasets/{dataset_name}/audio/raw/{surah_num}"

    # Find all wav files
    input_files = sorted(glob.glob(os.path.join(input_dir, f"{surah_part}-*.wav")))

    if not input_files:
        print(f"❌ Error: No audio files found in {input_dir}/{surah_part}-*.wav")
        return 0, 0, 1  # Return error count

    print(f"Found {len(input_files)} audio files to convert")

    # Determine output directory (mic quality)
    if '-' in surah_part and len(surah_part.split('-')) > 1 and surah_part.split('-')[1]:
        output_dir = f"../datasets/{dataset_name}/audio/mic/{surah_num}/{surah_part}"
    else:
        output_dir = f"../datasets/{dataset_name}/audio/mic/{surah_num}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert each file
    converted = 0
    skipped = 0
    errors = 0

    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)

        # Skip if already exists
        if os.path.exists(output_file):
            skipped += 1
            continue

        try:
            print(f"Converting: {filename}...", end=" ", flush=True)
            convert_to_mic_quality(input_file, output_file, target_sample_rate=8000)
            print("✓")
            converted += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            errors += 1

    print(f"Output directory: {output_dir}")
    return converted, skipped, errors

if __name__ == "__main__":
    main()
