#!/usr/bin/env python3
"""
Generate correct Quran text from Tanzil database instead of transcription
Usage: python3 generate_quran_text.py <segment_name>
       python3 generate_quran_text.py 001
       python3 generate_quran_text.py 002
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

def generate_text_for_surah(surah_num):
    """Generate text file based on Quran database"""
    if surah_num not in QURAN_TEXT:
        print(f"❌ Surah {surah_num:03d} not in database yet. Please add it to QURAN_TEXT.")
        return None

    if surah_num not in SEGMENT_MAPPING:
        print(f"❌ Segment mapping for surah {surah_num:03d} not defined. Please add it to SEGMENT_MAPPING.")
        return None

    ayahs = QURAN_TEXT[surah_num]
    mapping = SEGMENT_MAPPING[surah_num]

    lines = []
    for ayah_group in mapping:
        # Combine ayahs in this group
        text = " ".join(ayahs[i] for i in ayah_group)
        lines.append(text)

    return "\n".join(lines) + "\n"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 generate_quran_text.py <segment_name>")
        print("Examples:")
        print("  python3 generate_quran_text.py 001")
        print("  python3 generate_quran_text.py 002")
        sys.exit(1)

    segment_name = sys.argv[1]

    # Extract surah number (e.g., "001" -> 1, "002-04" -> 2)
    surah_num = int(segment_name.split('-')[0])

    # Get dataset name (default to Quran-A)
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "Quran-A"

    # Generate text
    print(f"Generating Quran text for surah {surah_num:03d}...")
    text_content = generate_text_for_surah(surah_num)

    if text_content is None:
        sys.exit(1)

    # Output file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, f"../datasets/{dataset_name}/text", f"{segment_name}.txt")

    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"✓ Saved to: {output_file}")
    print(f"\nContent:")
    print(text_content)

if __name__ == "__main__":
    main()
