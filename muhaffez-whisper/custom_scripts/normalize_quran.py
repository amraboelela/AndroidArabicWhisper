#!/usr/bin/env python3
import re

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
    input_file = "quran-simple-min.txt"
    output_file = "quran-simple-norm.txt"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = [normalize_arabic(line) for line in lines]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned {len(cleaned_lines)} lines")
    print(f"Normalized text saved to {output_file}")

if __name__ == "__main__":
    main()
