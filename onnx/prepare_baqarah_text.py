#!/usr/bin/env python3
"""
Extract the text for 002-01.wav (first 15 minutes of Al-Baqarah)
"""

# Read the quran file and get lines starting from Al-Baqarah
with open("quran-simple-min.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Al-Fatiha is lines 1-8 (with line 9 being empty)
# Al-Baqarah starts at line 10 (بسم الله...)
# Then line 11 is الم (Alif Lam Meem)
# We need to figure out how many ayahs fit in 15 minutes

# For now, let's extract the first 50 ayahs of Al-Baqarah
# (roughly estimating ~18 seconds per ayah on average)

baqarah_start = 10  # Line index (0-based would be 9, but file is 1-indexed)
num_ayahs = 50  # First 50 ayahs

# Get the text
baqarah_text_lines = []
for i in range(baqarah_start, min(baqarah_start + num_ayahs, len(lines))):
    line = lines[i].strip()
    if line and line != "-":  # Skip empty lines and separator
        baqarah_text_lines.append(line)

# Join all ayahs
baqarah_text = " ".join(baqarah_text_lines)

print(f"Extracted {len(baqarah_text_lines)} ayahs from Al-Baqarah")
print(f"Total characters: {len(baqarah_text)}")
print(f"\nFirst 200 characters:")
print(baqarah_text[:200])
print(f"\nLast 200 characters:")
print(baqarah_text[-200:])

# Save to file
with open("baqarah_002-01_text.txt", "w", encoding="utf-8") as f:
    f.write(baqarah_text)

print(f"\nSaved to: baqarah_002-01_text.txt")
