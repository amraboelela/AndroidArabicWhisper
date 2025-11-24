#!/usr/bin/env python3
"""
Reorganize augmented audio files into subdirectories by part.
Converts flat structure to hierarchical structure like raw audio.
"""

import os
import shutil
import re
from pathlib import Path

def reorganize_augmented_directory(base_dir):
    """
    Reorganize files in a directory into subdirectories based on part name.
    Example: 002-01-05.wav -> 002-01/002-01-05.wav
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return

    # Get all wav files in the directory (not in subdirectories)
    wav_files = [f for f in base_path.iterdir() if f.is_file() and f.suffix == '.wav']

    if not wav_files:
        print(f"No wav files found in {base_dir}")
        return

    print(f"\nProcessing: {base_dir}")
    print(f"Found {len(wav_files)} files to reorganize")

    moved_count = 0
    for wav_file in wav_files:
        # Extract part name (e.g., 002-01 from 002-01-05.wav)
        match = re.match(r'(002-\d+)-', wav_file.name)
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
            print(f"Warning: Could not extract part name from {wav_file.name}")

    print(f"✓ Moved {moved_count} files into subdirectories")

def main():
    dataset_path = Path(__file__).parent.parent / "datasets" / "Quran-A" / "audio" / "augmented"

    # List of all augmentation directories to reorganize
    augmentation_paths = [
        dataset_path / "pitch" / "plus4" / "002",
        dataset_path / "pitch" / "plus2" / "002",
        dataset_path / "pitch" / "minus2" / "002",
        dataset_path / "pitch" / "minus4" / "002",
        dataset_path / "speed" / "plus10" / "002",
        dataset_path / "speed" / "plus20" / "002",
        dataset_path / "speed" / "minus10" / "002",
        dataset_path / "speed" / "minus20" / "002",
    ]

    print("=" * 60)
    print("REORGANIZING AUGMENTED AUDIO FILES")
    print("=" * 60)

    for aug_path in augmentation_paths:
        reorganize_augmented_directory(aug_path)

    print("\n" + "=" * 60)
    print("✓ REORGANIZATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
