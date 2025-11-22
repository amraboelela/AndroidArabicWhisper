#!/usr/bin/env python3
"""
One-time script to split quran-simple-norm.txt into individual surah files
Creates text/001.txt, text/002.txt, etc.

File format:
- Lines with text: ayah content
- '-': surah boundary marker
- '*': page boundary marker
- Empty lines: page breaks (ignore)

Usage: python3 generate_surah_files.py
"""
import os

# Input file
QURAN_FILE = "quran-simple-norm.txt"

# Output directory
OUTPUT_DIR = "text"

def main():
    if not os.path.exists(QURAN_FILE):
        print(f"❌ Input file not found: {QURAN_FILE}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading {QURAN_FILE}...")

    with open(QURAN_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_surah = 1
    current_ayahs = []

    for line in lines:
        line = line.strip()

        # Skip isti'adha (first line)
        if line.startswith("اعوذ"):
            continue

        # '-' = surah boundary
        if line == '-':
            if current_ayahs:
                # Write surah to file
                output_file = os.path.join(OUTPUT_DIR, f"{current_surah:03d}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    for ayah in current_ayahs:
                        f.write(ayah + '\n')
                print(f"✓ Created {output_file} ({len(current_ayahs)} lines)")

                current_surah += 1
                current_ayahs = []
            continue

        # Skip empty lines and page markers (*)
        if not line or line == '*':
            continue

        # Regular ayah text
        current_ayahs.append(line)

    # Handle last surah (if no '-' at end)
    if current_ayahs:
        output_file = os.path.join(OUTPUT_DIR, f"{current_surah:03d}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for ayah in current_ayahs:
                f.write(ayah + '\n')
        print(f"✓ Created {output_file} ({len(current_ayahs)} lines)")

    print(f"\n{'='*60}")
    print(f"✓ Successfully generated {current_surah} surah files in {OUTPUT_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
