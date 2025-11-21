#!/usr/bin/env python3
"""
Fix transcribed Quran text by replacing with correct text from database
This runs after transcription to correct any errors
Usage: python3 fix_quran_text.py <dataset_name> <segment_name>
       python3 fix_quran_text.py Quran-A 001
       python3 fix_quran_text.py Quran-A 002
"""
import os
import sys

# Quran text database (Tanzil simple clean text)
# Format: surah_number -> list of ayahs
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

# Mapping of how ayahs are grouped into audio segments for each surah
# Format: surah_number -> list of ayah groups (each group becomes one line in output)
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

def fix_text_for_surah(surah_num, text_file):
    """Fix transcribed text by replacing with correct Quran text"""
    if surah_num not in QURAN_TEXT:
        print(f"⚠️  Surah {surah_num:03d} not in database - keeping transcribed text")
        return False

    if surah_num not in SEGMENT_MAPPING:
        print(f"⚠️  Segment mapping for surah {surah_num:03d} not defined - keeping transcribed text")
        return False

    # Read existing (transcribed) text
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as f:
            old_text = f.read()
        print(f"📄 Original transcribed text:")
        print(old_text)
        print("-" * 60)

    # Generate correct text
    ayahs = QURAN_TEXT[surah_num]
    mapping = SEGMENT_MAPPING[surah_num]

    lines = []
    for ayah_group in mapping:
        # Combine ayahs in this group
        text = " ".join(ayahs[i] for i in ayah_group)
        lines.append(text)

    correct_text = "\n".join(lines) + "\n"

    # Write corrected text
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(correct_text)

    print(f"✅ Fixed text:")
    print(correct_text)

    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 fix_quran_text.py <dataset_name> <segment_name>")
        print("Examples:")
        print("  python3 fix_quran_text.py Quran-A 001")
        print("  python3 fix_quran_text.py Quran-A 002")
        sys.exit(1)

    dataset_name = sys.argv[1]
    segment_name = sys.argv[2]

    # Extract surah number (e.g., "001" -> 1, "002-04" -> 2)
    surah_num = int(segment_name.split('-')[0])

    # Text file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    text_file = os.path.join(script_dir, f"../datasets/{dataset_name}/text/{segment_name}.txt")

    if not os.path.exists(text_file):
        print(f"❌ Text file not found: {text_file}")
        sys.exit(1)

    print(f"Fixing Quran text for surah {surah_num:03d}...")
    print("=" * 60)

    if fix_text_for_surah(surah_num, text_file):
        print("=" * 60)
        print(f"✓ Text corrected and saved to: {text_file}")
    else:
        print("=" * 60)
        print(f"⚠️  Could not fix text - manual correction needed")

if __name__ == "__main__":
    main()
