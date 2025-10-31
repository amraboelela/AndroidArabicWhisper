#!/usr/bin/env python3
"""Count words in Baqarah 002-01 text"""

# Read lines 9-56 (0-indexed: 8-55)
with open('quran-simple-norm.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Get lines, skip empty and separators
text_lines = []
for i in range(9, 56):  # Lines 10-56 in 1-indexed
    line = lines[i].strip()
    if line and line != '-' and line != '*':
        text_lines.append(line)

# Add Bismillah at the beginning
full_text = "بسم الله الرحمن الرحيم " + " ".join(text_lines)

words = full_text.split()
print(f"Total words: {len(words)}")
print(f"\nFirst 200 characters:")
print(full_text[:200])
print(f"\nLast 200 characters:")
print(full_text[-200:])

# Save
with open("baqarah_002-01_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"\nSaved to: baqarah_002-01_text.txt")
