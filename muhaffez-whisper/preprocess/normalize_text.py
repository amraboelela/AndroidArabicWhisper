#!/usr/bin/env python3
"""
Normalize Arabic Quran text
- Remove tashkeel (diacritics)
- Normalize hamza variants
Usage: python3 normalize_text.py <dataset_name> <segment_name>
       python3 normalize_text.py Quran-A 002-04
       python3 normalize_text.py Quran-A 001
"""
import re
import sys
import os

def remove_tashkeel(text):
    """Remove Arabic diacritics (tashkeel)"""
    # Harakat: fatha, damma, kasra, sukun, shadda, etc. (U+064B to U+065F)
    # Plus dagger alif (U+0670)
    tashkeel_pattern = r'[\u064B-\u065F\u0670]'
    return re.sub(tashkeel_pattern, '', text)

def remove_control_characters(text):
    """Remove control characters (Unicode category Cf)"""
    # Unicode control characters range (U+200B to U+200F, U+202A to U+202E, etc.)
    control_chars = r'[\u200B-\u200F\u202A-\u202E\u2060-\u2069\uFEFF]'
    return re.sub(control_chars, '', text)

def normalize_arabic(text):
    """Normalize Arabic text by removing tashkeel and normalizing hamza variants"""
    # 1. Remove diacritics (tashkeel) and control characters
    text = remove_tashkeel(text)
    text = remove_control_characters(text)

    # 2. Normalize hamza variants
    # Note: We keep ئ (yeh with hamza) and ؤ (waw with hamza) as is
    # because they represent distinct sounds and changing them would alter word meanings
    hamza_map = {
        'إ': 'ا',  # alif with hamza below
        'أ': 'ا',  # alif with hamza above
        'آ': 'ا',  # alif with madda
        # 'ؤ': 'و' - NOT normalized, kept as is
        # 'ئ': 'ي' - NOT normalized, kept as is
    }

    for old_char, new_char in hamza_map.items():
        text = text.replace(old_char, new_char)

    return text

def main():
    # Get dataset name and segment name from command line
    if len(sys.argv) < 3:
        print("Usage: python3 normalize_text.py <dataset_name> <segment_name>")
        print("Examples:")
        print("  python3 normalize_text.py Quran-A 002-04")
        print("  python3 normalize_text.py Quran-A 001")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_name = sys.argv[2]

    # Determine file paths
    input_file = f"../datasets/{dataset_name}/text/{segment_name}.txt"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Read transcribed text
    with open(input_file, "r", encoding="utf-8") as f:
        transcribed_lines = f.readlines()

    # Normalize text
    print(f"Normalizing: {input_file}")
    lines = [line.strip() for line in transcribed_lines if line.strip()]
    normalized_lines = [normalize_arabic(line) + "\n" for line in lines]

    with open(input_file, "w", encoding="utf-8") as f:
        f.writelines(normalized_lines)

    print(f"✓ Normalized {len(normalized_lines)} lines")
    print(f"✓ Saved to: {input_file}")

if __name__ == "__main__":
    main()
