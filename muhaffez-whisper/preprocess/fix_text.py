#!/usr/bin/env python3
"""
Fix and normalize Arabic Quran text
1. Fix transcribed text with correct Quran text from database
2. Normalize by removing tashkeel and normalizing hamza variants
Usage: python3 fix_text.py <dataset_name> <segment_name>
       python3 fix_text.py Quran-A 002-04
       python3 fix_text.py Quran-A 001
"""
import re
import sys
import os

# Quran text database (Tanzil simple clean text)
QURAN_TEXT = {
    1: [  # Al-Fatiha
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "الرحمن الرحيم",
        "مالك يوم الدين",
        "إياك نعبد وإياك نستعين",
        "اهدنا الصراط المستقيم",
        "صراط الذين أنعمت عليهم غير المغضوب عليهم ولا الضالين"
    ],
    # Add more surahs as needed
}

# Mapping of how ayahs are grouped into audio segments
SEGMENT_MAPPING = {
    1: [  # Al-Fatiha - 6 segments
        [0],        # Basmalah
        [1, 2],     # Al-hamdu + Ar-rahman
        [3],        # Maliki yawm
        [4],        # Iyyaka na'budu
        [5],        # Ihdina
        [6]         # Sirat alladhina
    ],
}

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

def fix_quran_text(surah_num):
    """Get correct Quran text from database if available"""
    if surah_num not in QURAN_TEXT or surah_num not in SEGMENT_MAPPING:
        return None

    ayahs = QURAN_TEXT[surah_num]
    mapping = SEGMENT_MAPPING[surah_num]

    lines = []
    for ayah_group in mapping:
        text = " ".join(ayahs[i] for i in ayah_group)
        lines.append(text + "\n")

    return lines

def main():
    # Get dataset name and segment name from command line
    if len(sys.argv) < 3:
        print("Usage: python3 fix_text.py <dataset_name> <segment_name>")
        print("Examples:")
        print("  python3 fix_text.py Quran-A 002-04")
        print("  python3 fix_text.py Quran-A 001")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_name = sys.argv[2]

    # Extract surah number
    surah_num = int(segment_name.split('-')[0])

    # Determine file paths
    input_file = f"../datasets/{dataset_name}/text/{segment_name}.txt"
    output_file = f"../datasets/{dataset_name}/text/{segment_name}.txt"  # Overwrite the same file

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    # Step 1: Try to fix with correct Quran text
    fixed_lines = fix_quran_text(surah_num)

    if fixed_lines:
        print(f"✓ Using correct Quran text for surah {surah_num:03d}")
        lines = fixed_lines
    else:
        print(f"⚠️  Surah {surah_num:03d} not in database - using transcribed text")
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

    # Step 2: Normalize text
    print(f"Normalizing: {input_file}")
    cleaned_lines = [normalize_arabic(line) for line in lines]

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"✓ Normalized {len(cleaned_lines)} lines")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    main()
