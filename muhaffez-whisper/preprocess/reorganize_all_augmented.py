#!/usr/bin/env python3
"""
Reorganize all augmented audio files into subdirectories by part.
Works for all surahs and all augmentation types (pitch and speed).
"""

import os
import shutil
import re
from pathlib import Path

def reorganize_augmented_directory(base_dir):
    """
    Reorganize files in a directory into subdirectories based on part name.
    Example: 002-01-05.wav -> 002-01/002-01-05.wav
    Only applies to files with part structure (XXX-YY-ZZ.wav)
    Skips files without parts (XXX-YY.wav for single-part surahs like 001)
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return

    # Get all wav files in the directory (not in subdirectories)
    wav_files = [f for f in base_path.iterdir() if f.is_file() and f.suffix == '.wav']

    if not wav_files:
        return

    print(f"\nProcessing: {base_dir}")
    print(f"Found {len(wav_files)} files")

    moved_count = 0
    skipped_count = 0

    for wav_file in wav_files:
        # Extract part name (e.g., 002-01 from 002-01-05.wav)
        # This pattern requires 3 segments: XXX-YY-ZZ
        match = re.match(r'(\d{3}-\d{2})-\d+\.wav$', wav_file.name)
        if match:
            part_name = match.group(1)
            part_dir = base_path / part_name

            # Create subdirectory if it doesn't exist
            part_dir.mkdir(exist_ok=True)

            # Move file to subdirectory
            dest_file = part_dir / wav_file.name
            shutil.move(str(wav_file), str(dest_file))
            moved_count += 1
        else:
            # Skip files that don't have part structure (e.g., 001-XX.wav)
            skipped_count += 1

    if moved_count > 0:
        print(f"✓ Moved {moved_count} files into subdirectories")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} files (no part structure)")

def main():
    dataset_path = Path(__file__).parent.parent / "datasets" / "Quran-A" / "audio" / "augmented"

    print("=" * 60)
    print("REORGANIZING ALL AUGMENTED AUDIO FILES")
    print("=" * 60)

    # Process all pitch augmentations
    pitch_variations = ["plus4", "plus2", "minus2", "minus4"]
    for variation in pitch_variations:
        pitch_dir = dataset_path / "pitch" / variation
        if pitch_dir.exists():
            print(f"\n--- Processing pitch/{variation} ---")
            # Process all surahs in this variation
            for surah_dir in sorted(pitch_dir.iterdir()):
                if surah_dir.is_dir():
                    reorganize_augmented_directory(surah_dir)

    # Process all speed augmentations
    speed_variations = ["plus10", "plus20", "minus10", "minus20"]
    for variation in speed_variations:
        speed_dir = dataset_path / "speed" / variation
        if speed_dir.exists():
            print(f"\n--- Processing speed/{variation} ---")
            # Process all surahs in this variation
            for surah_dir in sorted(speed_dir.iterdir()):
                if surah_dir.is_dir():
                    reorganize_augmented_directory(surah_dir)

    print("\n" + "=" * 60)
    print("✓ REORGANIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
